from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
import numpy as np
import sounddevice as sd

log = logging.getLogger("oxmoira.audio")

TARGET_SR = 16000
FRAME_MS = 30
FRAME_SAMPLES = TARGET_SR * FRAME_MS // 1000  # 480


@dataclass
class AudioConfig:
    device_hint: str = os.getenv("OXMOIRA_AUDIO_DEVICE", "BlackHole 2ch")
    asr_base_url: str = os.getenv("OXMOIRA_ASR_URL", "http://127.0.0.1:8000/v1/audio/transcriptions")
    asr_api_key: str = os.getenv("OXMOIRA_ASR_KEY", "")
    asr_model: str = os.getenv("OXMOIRA_ASR_MODEL", "whisper-1")
    asr_language: str = os.getenv("OXMOIRA_ASR_LANG", "")
    asr_timeout_s: float = float(os.getenv("OXMOIRA_ASR_TIMEOUT", "15"))
    asr_max_inflight: int = int(os.getenv("OXMOIRA_ASR_INFLIGHT", "3"))
    # Endpointing
    silence_end_ms: int = 450
    min_utterance_ms: int = 350
    max_utterance_ms: int = 8000
    preroll_ms: int = 300
    abs_min_rms: float = 0.006  # float32 full-scale RMS floor
    speech_over_noise: float = 3.0


@dataclass
class Utterance:
    seq: int
    speaker_id: str
    text: str
    t_start: float
    t_end: float
    asr_latency_s: float
    extra: Dict[str, Any] = field(default_factory=dict)


def _resolve_device(hint: str) -> Optional[int]:
    hint_l = hint.lower().strip()
    if hint_l.isdigit():
        return int(hint_l)
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and hint_l in dev["name"].lower():
            return idx
    log.warning("Input device '%s' not found; using system default input.", hint)
    return None


def _to_mono_16k(block: np.ndarray, src_sr: int) -> np.ndarray:
    mono = block.mean(axis=1) if block.ndim == 2 else block
    if src_sr == TARGET_SR:
        return mono.astype(np.float32, copy=False)
    n_out = int(round(len(mono) * TARGET_SR / src_sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, mono).astype(np.float32)


def _encode_wav(pcm16: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


class AudioEngine:
    """
    sounddevice callback thread -> (call_soon_threadsafe) -> asyncio raw queue
    -> energy-endpointing chunker -> ordered concurrent ASR -> Utterance queue.

    The PortAudio callback does a single array copy and a thread-safe enqueue; nothing
    else runs on the realtime thread.
    """

    def __init__(self, cfg: Optional[AudioConfig] = None,
                 transcriber: Optional[Callable[[int, bytes], Awaitable[Any]]] = None) -> None:
        """`transcriber(seq, wav)` replaces the HTTP endpoint (e.g. GeminiTranscriber); it returns an ASR payload."""
        self.cfg = cfg or AudioConfig()
        self._transcriber = transcriber
        self.last_error = ""
        self.utterances: "asyncio.Queue[Utterance]" = asyncio.Queue()
        self.level: float = 0.0  # latest RMS, readable by the UI
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._raw: "asyncio.Queue[np.ndarray]" = asyncio.Queue(maxsize=512)
        self._pending: "asyncio.Queue[Optional[asyncio.Task]]" = asyncio.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._tasks: List[asyncio.Task] = []
        self._client: Optional[httpx.AsyncClient] = None
        self._sem = asyncio.Semaphore(self.cfg.asr_max_inflight)
        self._src_sr = TARGET_SR
        self._seq = 0
        self._t0 = 0.0
        self._dropped = 0

    # ---- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._t0 = time.monotonic()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.cfg.asr_timeout_s),
            limits=httpx.Limits(max_connections=self.cfg.asr_max_inflight + 1,
                                max_keepalive_connections=self.cfg.asr_max_inflight + 1),
        )
        device = _resolve_device(self.cfg.device_hint)
        info = sd.query_devices(device, "input")
        channels = max(1, min(2, int(info["max_input_channels"])))
        try:
            self._open_stream(device, channels, TARGET_SR)
        except sd.PortAudioError:
            sr = int(info["default_samplerate"])
            log.warning("16 kHz unsupported on '%s'; capturing at %d Hz and resampling.", info["name"], sr)
            self._open_stream(device, channels, sr)
        self._stream.start()
        log.info("Capturing from '%s' (%d ch @ %d Hz)", info["name"], channels, self._src_sr)
        self._tasks = [
            asyncio.create_task(self._chunker(), name="oxmoira-chunker"),
            asyncio.create_task(self._emitter(), name="oxmoira-emitter"),
        ]

    def _open_stream(self, device: Optional[int], channels: int, sr: int) -> None:
        self._src_sr = sr
        blocksize = sr * FRAME_MS // 1000
        self._stream = sd.InputStream(
            device=device, channels=channels, samplerate=sr, blocksize=blocksize,
            dtype="float32", latency="low", callback=self._callback,
        )

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ---- realtime thread -------------------------------------------------
    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            log.debug("PortAudio status: %s", status)
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._enqueue_raw, indata.copy())

    def _enqueue_raw(self, block: np.ndarray) -> None:
        try:
            self._raw.put_nowait(block)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 50 == 1:
                log.error("Raw audio queue overflow; dropped %d blocks (chunker starved).", self._dropped)

    # ---- chunker ---------------------------------------------------------
    async def _chunker(self) -> None:
        cfg = self.cfg
        preroll_frames = max(1, cfg.preroll_ms // FRAME_MS)
        silence_frames_end = max(1, cfg.silence_end_ms // FRAME_MS)
        min_frames = max(1, cfg.min_utterance_ms // FRAME_MS)
        max_frames = max(2, cfg.max_utterance_ms // FRAME_MS)

        ring: List[np.ndarray] = []
        active: List[np.ndarray] = []
        in_speech = False
        silence_run = 0
        voiced = 0
        noise_floor = cfg.abs_min_rms
        t_start = 0.0
        carry = np.zeros(0, dtype=np.float32)

        while True:
            block = await self._raw.get()
            mono = _to_mono_16k(block, self._src_sr)
            if carry.size:
                mono = np.concatenate((carry, mono))
            n_full = len(mono) // FRAME_SAMPLES
            carry = mono[n_full * FRAME_SAMPLES:]
            for i in range(n_full):
                frame = mono[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
                rms = float(np.sqrt(np.mean(frame * frame)) + 1e-12)
                self.level = rms
                threshold = max(cfg.abs_min_rms, noise_floor * cfg.speech_over_noise)
                is_speech = rms >= threshold

                if not in_speech:
                    noise_floor = 0.95 * noise_floor + 0.05 * rms if not is_speech else noise_floor
                    ring.append(frame)
                    if len(ring) > preroll_frames:
                        ring.pop(0)
                    if is_speech:
                        in_speech = True
                        active = list(ring)
                        ring = []
                        silence_run = 0
                        voiced = 1
                        t_start = time.monotonic() - self._t0 - len(active) * FRAME_MS / 1000.0
                    continue

                active.append(frame)
                if is_speech:
                    silence_run = 0
                    voiced += 1
                else:
                    silence_run += 1

                ended = silence_run >= silence_frames_end
                forced = len(active) >= max_frames
                if ended or forced:
                    if voiced >= min_frames:
                        t_end = time.monotonic() - self._t0
                        audio = np.concatenate(active)
                        self._submit(audio, t_start, t_end)
                    in_speech = False
                    active = []
                    silence_run = 0
                    voiced = 0

    # ---- ASR (ordered, bounded concurrency) -------------------------------
    def _submit(self, audio: np.ndarray, t_start: float, t_end: float) -> None:
        seq = self._seq
        self._seq += 1
        pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        wav = _encode_wav(pcm16)
        task = asyncio.create_task(self._transcribe(seq, wav, t_start, t_end), name=f"asr-{seq}")
        self._pending.put_nowait(task)

    async def _transcribe(self, seq: int, wav: bytes, t_start: float, t_end: float) -> List[Utterance]:
        if self._transcriber is not None:
            async with self._sem:
                t0 = time.monotonic()
                try:
                    payload = await self._transcriber(seq, wav)
                    self.last_error = ""
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    log.error("Transcription failed for chunk %d: %s", seq, exc)
                    return []
                latency = time.monotonic() - t0
            return self._parse(seq, payload, t_start, t_end, latency)
        assert self._client is not None
        headers = {"Authorization": f"Bearer {self.cfg.asr_api_key}"} if self.cfg.asr_api_key else {}
        data = {"model": self.cfg.asr_model, "response_format": "verbose_json", "diarize": "true"}
        if self.cfg.asr_language:
            data["language"] = self.cfg.asr_language
        files = {"file": (f"chunk_{seq}.wav", wav, "audio/wav")}
        async with self._sem:
            t0 = time.monotonic()
            last_exc: Optional[Exception] = None
            for attempt in range(2):
                try:
                    resp = await self._client.post(self.cfg.asr_base_url, headers=headers, data=data, files=files)
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    last_exc = exc
                    await asyncio.sleep(0.25 * (attempt + 1))
            else:
                self.last_error = f"ASR endpoint: {last_exc}"
                log.error("ASR failed for chunk %d: %s", seq, last_exc)
                return []
            latency = time.monotonic() - t0
        return self._parse(seq, payload, t_start, t_end, latency)

    @staticmethod
    def _parse(seq: int, payload: Any, t_start: float, t_end: float, latency: float) -> List[Utterance]:
        """
        Accepts any of:
          {"segments":   [{"speaker": "...", "text": "...", "start": f, "end": f}, ...]}
          {"utterances": [{"speaker": "...", "text": "..."}, ...]}
          {"text": "..."}                       (no diarization -> speaker_id 'UNKNOWN')
        Consecutive segments from the same speaker are merged.
        """
        if not isinstance(payload, dict):
            return []
        segs = payload.get("segments") or payload.get("utterances")
        out: List[Utterance] = []
        if isinstance(segs, list) and segs:
            merged: List[Dict[str, Any]] = []
            for s in segs:
                if not isinstance(s, dict):
                    continue
                text = str(s.get("text", "")).strip()
                if not text:
                    continue
                spk = str(s.get("speaker") or s.get("speaker_id") or s.get("speaker_label") or "UNKNOWN")
                if merged and merged[-1]["speaker"] == spk:
                    merged[-1]["text"] += " " + text
                else:
                    merged.append({"speaker": spk, "text": text})
            for m in merged:
                out.append(Utterance(seq, m["speaker"], m["text"], t_start, t_end, latency))
        else:
            text = str(payload.get("text", "")).strip()
            if text:
                spk = str(payload.get("speaker") or payload.get("speaker_id") or "UNKNOWN")
                extra = {k: payload[k] for k in ("utterance_id", "duration_ms") if k in payload}
                out.append(Utterance(seq, spk, text, t_start, t_end, latency, extra))
        return out

    async def _emitter(self) -> None:
        while True:
            task = await self._pending.get()
            if task is None:
                return
            try:
                for utt in await task:
                    await self.utterances.put(utt)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("ASR task crashed")


async def _demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    engine = AudioEngine()
    await engine.start()
    try:
        while True:
            u = await engine.utterances.get()
            print(f"[{u.t_start:7.2f}-{u.t_end:7.2f}] asr={u.asr_latency_s:.2f}s {u.speaker_id}: {u.text}")
    finally:
        await engine.stop()


if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass
