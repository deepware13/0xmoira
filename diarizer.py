"""Automatic speaker detection from the voice (no API key needed).

The browser streams the microphone (16 kHz mono int16) to the app while its own speech recognition produces
the text. This module
  1. cuts the audio stream into voiced segments (energy VAD with an adaptive noise floor),
  2. computes a voice print per segment (WeSpeaker ResNet34-LM, CC BY 4.0, ONNX on CPU),
  3. re-clusters all segments of the session (agglomerative, average linkage, cosine) after every new one,
     keeping labels stable (Hungarian matching on speaking time) and reporting earlier segments whose speaker
     changed, so early mistakes are corrected,
  4. assigns each finished transcript line to the speaker who was talking during the time it covers.

Clustering design and thresholds follow the 0xmoira prototype's diarizer (calibrated there on simulated room
conversations: same speaker cosine ~0.68, different speakers ~0.17).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np

from providers import CRED_DIR

log = logging.getLogger("oxmoira.diarizer")

SR = 16000
MODEL_REV = "f0c48c298fd835726c27956a5d617bad7115627e"
MODEL_URL = (f"https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/{MODEL_REV}/"
             "voxceleb_resnet34_LM.onnx")
MODEL_SHA256 = "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"
MODEL_PATH = CRED_DIR / "models" / "voxceleb_resnet34_LM.onnx"

# ResNet34 in simulated rooms (RT60 0.3-0.8 s, SNR 8-20 dB, 1.5-3.5 s segments): same speaker 0.52 +- 0.12,
# different speakers 0.16 +- 0.11. 0.41 split 4 voices into 7. Streaming tests with constant room noise: 0.32 merged
# two similar voices in 4/4 runs, 0.36 in 3/4, 0.38 in 1/4; none over-split. 0.36 errs towards fewer speakers.
SIM_THRESHOLD = 0.36      # min average-linkage cosine similarity to be the same speaker
MIN_RELIABLE_S = 1.5      # shorter segments never found a speaker of their own
MIN_CLUSTER_S = 5.0       # a new voice needs this much speech before it becomes a speaker of its own
HYSTERESIS = 0.08         # an unsure segment switches speaker only if clearly closer to another
WINDOW = 500              # segments re-clustered (older ones keep their label)

FRAME = 480               # 30 ms
SILENCE_END_MS = 250
MAX_SEGMENT_S = 6.0
MIN_SEGMENT_S = 0.4
PREROLL_FRAMES = 6


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def kaldi_mel_banks(num_bins: int = 80, n_fft: int = 512, sr: int = SR, low: float = 20.0) -> np.ndarray:
    """Kaldi-style triangular mel filterbank (mel = 1127 ln(1 + f/700)), shape (n_fft/2+1, num_bins)."""
    def mel(f):
        return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)
    high = sr / 2
    mel_lo, mel_hi = mel(low), mel(high)
    delta = (mel_hi - mel_lo) / (num_bins + 1)
    fft_mel = mel(np.arange(n_fft // 2 + 1) * sr / n_fft)
    banks = np.zeros((n_fft // 2 + 1, num_bins), dtype=np.float32)
    for b in range(num_bins):
        left, center, right = mel_lo + b * delta, mel_lo + (b + 1) * delta, mel_lo + (b + 2) * delta
        up = (fft_mel - left) / (center - left)
        down = (right - fft_mel) / (right - center)
        banks[:, b] = np.maximum(0.0, np.minimum(up, down))
    return banks


async def ensure_model(progress: Optional[Callable[[str], None]] = None) -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".part")
    if progress:
        progress("DOWNLOADING VOICE MODEL (26 MB, ONCE)…")
    h = hashlib.sha256()
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        async with client.stream("GET", MODEL_URL) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                async for chunk in resp.aiter_bytes(1 << 16):
                    fh.write(chunk)
                    h.update(chunk)
    if h.hexdigest() != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Voice model download is corrupted (checksum mismatch).")
    tmp.replace(MODEL_PATH)
    return MODEL_PATH


class Embedder:
    """WeSpeaker r-vector: Kaldi fbank (80 mel, 25/10 ms, Hamming, pre-emphasis) with mean normalisation."""
    n_fft, win, hop, preemph = 512, 400, 160, 0.97

    def __init__(self, path: Path) -> None:
        import onnxruntime as rt
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3
        self.session = rt.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.banks = kaldi_mel_banks()
        self.window = np.hamming(self.win).astype(np.float32)

    def fbank(self, x: np.ndarray) -> np.ndarray:
        frames = np.lib.stride_tricks.sliding_window_view(x.astype(np.float32), self.win)[:: self.hop]
        frames = frames - frames.mean(axis=-1, keepdims=True)
        frames = frames - self.preemph * np.concatenate([frames[:, :1], frames[:, :-1]], axis=-1)
        spec = np.abs(np.fft.rfft(frames * self.window, self.n_fft)).astype(np.float32) ** 2
        with np.errstate(all="ignore"):
            mel = spec @ self.banks
        feats = np.log(np.maximum(np.where(np.isfinite(mel), mel, 0.0), np.finfo(np.float32).eps))
        return feats - feats.mean(axis=0, keepdims=True)

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        # WeSpeaker expects int16-scaled amplitude
        feats = self.fbank(pcm * 32768.0)[None]
        (e,) = self.session.run(["embs"], {"feats": feats.astype(np.float32)})
        return _norm(e[0])


@dataclass
class Item:
    id: int
    emb: np.ndarray
    dur: float
    t0: float
    t1: float
    label: Optional[str] = None
    conf: int = 0


@dataclass
class VoiceDiarizer:
    items: List[Item] = field(default_factory=list)
    used: List[str] = field(default_factory=list)
    next_id: int = 0
    max_speakers: int = 0     # 0 = automatic
    def _new_label(self, taken: set) -> str:
        # Lowest free number: a provisional voice that was later merged away gives its number back,
        # so speakers are numbered 1, 2, 3 without gaps. Labels of segments outside the window stay reserved.
        frozen = {x.label for x in self.items[:-WINDOW] if x.label}
        k = 1
        while f"Speaker {k}" in taken or f"Speaker {k}" in frozen:
            k += 1
        return f"Speaker {k}"

    def add(self, emb: np.ndarray, t0: float, t1: float) -> Tuple[Item, List[Item]]:
        before = {it.id: it.label for it in self.items}
        it = Item(self.next_id, emb, t1 - t0, t0, t1)
        self.next_id += 1
        self.items.append(it)
        self._recluster()
        changed = [x for x in self.items if x.id in before and before[x.id] and before[x.id] != x.label]
        return it, changed

    def _recluster(self) -> None:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.optimize import linear_sum_assignment

        items = self.items[-WINDOW:]
        reliable = [x for x in items if x.dur >= MIN_RELIABLE_S] or [max(items, key=lambda x: x.dur)]
        X = np.stack([x.emb for x in reliable])
        if len(reliable) == 1:
            groups = np.array([1])
        else:
            with np.errstate(all="ignore"):
                Z = linkage(X, method="average", metric="cosine")
            groups = fcluster(Z, t=1.0 - SIM_THRESHOLD, criterion="distance")
            if self.max_speakers > 0 and groups.max() > self.max_speakers:
                groups = fcluster(Z, t=self.max_speakers, criterion="maxclust")
        durs = np.array([x.dur for x in reliable])
        secs = {g: float(durs[groups == g].sum()) for g in set(groups.tolist())}
        big = [g for g, v in secs.items() if v >= MIN_CLUSTER_S]
        provisional = np.zeros(len(reliable), dtype=bool)
        if big:
            bc = {g: _norm(X[groups == g].mean(0)) for g in big}
            for g, v in secs.items():
                if v < MIN_CLUSTER_S:
                    m = groups == g
                    groups[m] = max(big, key=lambda b: float(_norm(X[m].mean(0)) @ bc[b]))
                    provisional |= m
        ids = sorted(set(groups.tolist()))
        cents = np.stack([_norm(X[groups == g].mean(0)) for g in ids])
        E = np.stack([x.emb for x in items])
        with np.errstate(all="ignore"):
            S = E @ cents.T
        member = {id(x): ids.index(g) for x, g in zip(reliable, groups)}
        unsure = {id(x) for x, p in zip(reliable, provisional) if p}
        assign = np.array([member.get(id(x), int(np.argmax(S[i]))) for i, x in enumerate(items)])

        old = sorted({x.label for x in items if x.label})
        mapping: Dict[int, str] = {}
        if old:
            M = np.zeros((len(ids), len(old)))
            for i, x in enumerate(items):
                if x.label:
                    M[assign[i], old.index(x.label)] += x.dur
            for r, c in zip(*linear_sum_assignment(-M)):
                if M[r, c] > 0:
                    mapping[r] = old[c]
        for g in range(len(ids)):
            if g not in mapping:
                mapping[g] = self._new_label(set(mapping.values()))
        for lab in mapping.values():
            if lab not in self.used:
                self.used.append(lab)

        label_to_g = {lab: g for g, lab in mapping.items()}
        for i, x in enumerate(items):
            if (id(x) in member and id(x) not in unsure) or x.label not in label_to_g:
                continue
            prev = label_to_g[x.label]
            if assign[i] != prev and S[i, assign[i]] - S[i, prev] < HYSTERESIS:
                assign[i] = prev
        for i, x in enumerate(items):
            g = assign[i]
            other = float(np.max(np.delete(S[i], g))) if len(ids) > 1 else -1.0
            margin = float(np.clip((S[i, g] - other) / 0.4, 0.0, 1.0))
            x.label = mapping[g]
            x.conf = int(20 + 25 * margin) if (id(x) in unsure or x.dur < MIN_RELIABLE_S) else int(50 + 50 * margin)


class VoiceTracker:
    """Consumes the live PCM stream, keeps a timeline of voiced segments with speaker labels, and answers
    'who was speaking during the last N seconds'. Timestamps are this process's monotonic clock."""

    def __init__(self, embedder: Embedder, on_relabel: Optional[Callable[[List[Item]], None]] = None) -> None:
        self.embedder = embedder
        self.diar = VoiceDiarizer()
        self.on_relabel = on_relabel
        self.level = 0.0
        self.open_since: Optional[float] = None
        self._carry = np.zeros(0, dtype=np.float32)
        self._ring: List[np.ndarray] = []
        self._active: List[np.ndarray] = []
        self._rms_hist: List[float] = []   # last ~10 s of frame energies, for a robust noise floor
        self._silence = 0
        self._voiced = 0
        self._t_start = 0.0
        self._pending = 0
        self.last_label: Optional[str] = None

    @property
    def busy(self) -> bool:
        return self.open_since is not None or self._pending > 0

    def feed(self, pcm16: bytes) -> None:
        now = time.monotonic()
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if self._carry.size:
            x = np.concatenate((self._carry, x))
        n = len(x) // FRAME
        self._carry = x[n * FRAME:]
        for i in range(n):
            frame = x[i * FRAME:(i + 1) * FRAME]
            t = now - (n - i) * FRAME / SR
            rms = float(np.sqrt(np.mean(frame * frame)) + 1e-12)
            self.level = rms
            # Noise floor = 5th percentile of the last 10 s, updated on every frame. Unlike a floor that only
            # learns during silence, it cannot get stuck when the room is louder than the initial guess; the
            # long window keeps it from rising into the speech level during a long uninterrupted turn.
            self._rms_hist.append(rms)
            if len(self._rms_hist) > 333:
                self._rms_hist.pop(0)
            floor = float(np.percentile(self._rms_hist, 5)) if len(self._rms_hist) >= 10 else 0.002
            speech = rms >= max(0.004, floor * 2.5)
            if self.open_since is None:
                self._ring = (self._ring + [frame])[-PREROLL_FRAMES:]
                if speech:
                    self._active, self._ring = list(self._ring), []
                    self.open_since = t - len(self._active) * FRAME / SR
                    self._silence, self._voiced = 0, 1
                continue
            self._active.append(frame)
            if speech:
                self._silence = 0
                self._voiced += 1
            else:
                self._silence += 1
            dur = len(self._active) * FRAME / SR
            if self._silence * 30 >= SILENCE_END_MS or dur >= MAX_SEGMENT_S:
                self._close(t)

    def flush(self) -> None:
        if self.open_since is not None:
            self._close(time.monotonic())

    def _close(self, t_end: float) -> None:
        t0 = self.open_since or t_end
        audio = np.concatenate(self._active) if self._active else np.zeros(0, dtype=np.float32)
        voiced_s = self._voiced * FRAME / SR
        self.open_since, self._active = None, []
        if voiced_s < MIN_SEGMENT_S or audio.size < SR // 2:
            return
        self._pending += 1
        asyncio.get_running_loop().create_task(self._embed(audio, t0, t_end))

    async def _embed(self, audio: np.ndarray, t0: float, t1: float) -> None:
        try:
            emb = await asyncio.get_running_loop().run_in_executor(None, self.embedder.embed, audio)
            it, changed = self.diar.add(emb, t0, t1)
            self.last_label = it.label
            if changed and self.on_relabel:
                self.on_relabel(changed)
        except Exception:
            log.exception("voice embedding failed")
        finally:
            self._pending -= 1

    async def assign(self, t_final: float, words: int) -> Tuple[str, List[int]]:
        """Speaker for a transcript line that was finalised at `t_final` and has `words` words."""
        deadline = time.monotonic() + 1.2
        while self.busy and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        # The line was spoken roughly within [t_final - finalisation delay - speaking time, t_final].
        # Speaking time is estimated conservatively (fast speech, 4 words/s) so the window does not
        # reach back into the previous speaker's turn; only segments mostly inside it count.
        span = max(0.8, words / 4.0)
        lo, hi = t_final - span - 1.2, t_final + 0.3
        overlap: Dict[str, float] = {}
        ids: List[int] = []
        for it in self.diar.items[-60:]:
            o = min(hi, it.t1) - max(lo, it.t0)
            if it.label and o > 0 and (o >= 0.4 * it.dur or o >= 0.5):
                overlap[it.label] = overlap.get(it.label, 0.0) + o
                ids.append(it.id)
        if overlap:
            return max(overlap, key=overlap.get), ids
        return self.last_label or "Speaker 1", ids

    def set_max_speakers(self, n: int) -> None:
        self.diar.max_speakers = max(0, int(n))
        if self.diar.items:
            before = {x.id: x.label for x in self.diar.items}
            self.diar._recluster()
            changed = [x for x in self.diar.items if before.get(x.id) and before[x.id] != x.label]
            if changed and self.on_relabel:
                self.on_relabel(changed)

    def label_of(self, ids: Sequence[int]) -> Optional[str]:
        by_id = {it.id: it for it in self.diar.items[-WINDOW:]}
        tally: Dict[str, float] = {}
        for i in ids:
            it = by_id.get(i)
            if it and it.label:
                tally[it.label] = tally.get(it.label, 0.0) + it.dur
        return max(tally, key=tally.get) if tally else None
