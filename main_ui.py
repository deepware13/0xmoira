from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nicegui import app, ui

from agent_core import BUDGET_USD, SYNTH_EVERY, AgentCore, AnalysisResult
from audio_engine import AudioConfig, AudioEngine
from providers import (AnthropicProvider, GeminiProvider, GeminiTranscriber, LoginError, clear_credentials,
                       default_anthropic_model, default_gemini_model, load_credentials, mask_key,
                       save_credentials, verify_anthropic, verify_gemini)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("oxmoira.ui")

# =========================================================================== #
# Backend state (single local user; everything runs on NiceGUI's event loop)
# =========================================================================== #
SOURCES = {
    "browser": "BROWSER MIC",
    "gemini": "SERVER MIC · GEMINI ASR",
    "endpoint": "SERVER MIC · ASR ENDPOINT",
    "text": "TEXT",
}
LANGS = ["en-US", "en-GB", "de-DE", "de-AT", "de-CH"]


@dataclass
class Seg:
    uid: str
    speaker: str
    text: str
    t: float


@dataclass
class State:
    provider: Any = None
    gemini_asr: Optional[GeminiProvider] = None
    gemini_asr_model: str = ""
    agent: Optional[AgentCore] = None
    engine: Optional[AudioEngine] = None
    pump: Optional[asyncio.Task] = None
    feed_task: Optional[asyncio.Task] = None
    source: str = "browser"
    lang: str = "en-US"
    speaker: str = "A"
    recording: bool = False
    rec_client: Optional[str] = None
    segs: List[Seg] = field(default_factory=list)
    seq: int = 0
    t0: float = 0.0
    version: int = 0
    error: str = ""
    speaker_order: Dict[str, int] = field(default_factory=dict)


S = State()
PENDING: Dict[str, Dict[str, Any]] = {}   # per-client verified login data; keys never go back to the browser


def e(x: Any) -> str:
    return html.escape(str(x), quote=True)


def spk_index(label: str) -> int:
    if label not in S.speaker_order:
        S.speaker_order[label] = len(S.speaker_order)
    return S.speaker_order[label] % 6


def spk_short(label: str) -> str:
    m = re.fullmatch(r"(?i)speaker[ _-]?(\w+)", label.strip())
    if m:
        return m.group(1)[:2].upper()
    return label.strip()[:2].upper() or "?"


def badge(label: str, extra: str = "") -> str:
    return f'<span class="spk spk-{spk_index(label)} {extra}" title="{e(label)}">{e(spk_short(label))}</span>'


# =========================================================================== #
# Session control
# =========================================================================== #
def ingest(speaker: str, text: str, confidence: Optional[int] = None) -> None:
    text = text.strip()
    if not text or S.agent is None:
        return
    if not S.segs:
        S.t0 = time.time()
    S.seq += 1
    uid = f"u{S.seq}"
    S.segs.append(Seg(uid, speaker, text, time.time() - S.t0))
    spk_index(speaker)
    S.version += 1
    payload: Dict[str, Any] = {"utterance_id": uid, "speaker": speaker, "text": text,
                               "duration_ms": int(len(text.split()) / 2.5 * 1000)}
    if confidence is not None:
        payload["confidence"] = confidence
    S.agent.submit(payload)


async def _pump() -> None:
    assert S.engine is not None
    while True:
        utt = await S.engine.utterances.get()
        ingest(utt.speaker_id if utt.speaker_id != "UNKNOWN" else S.speaker, utt.text)


async def start_server_mic() -> None:
    if S.source == "gemini":
        if S.gemini_asr is None:
            raise RuntimeError("Gemini ASR needs a Gemini key: sign out and add one on the sign-in screen.")
        budget = S.agent.budget if S.agent else None
        transcriber = GeminiTranscriber(S.gemini_asr, S.gemini_asr_model,
                                        on_cost=(budget.add_cost if budget else None))
        S.engine = AudioEngine(AudioConfig(), transcriber=transcriber)
    else:
        S.engine = AudioEngine(AudioConfig())
    await S.engine.start()
    S.pump = asyncio.create_task(_pump(), name="oxmoira-pump")


async def stop_server_mic() -> None:
    if S.pump:
        S.pump.cancel()
        await asyncio.gather(S.pump, return_exceptions=True)
        S.pump = None
    if S.engine:
        await S.engine.stop()
        S.engine = None


async def stop_recording() -> None:
    if S.source in ("gemini", "endpoint"):
        await stop_server_mic()
    elif S.source == "browser":
        ui.run_javascript("oxm.speech.stop()")
    S.recording = False
    S.rec_client = None


async def stop_feed() -> None:
    if S.feed_task:
        S.feed_task.cancel()
        await asyncio.gather(S.feed_task, return_exceptions=True)
        S.feed_task = None


async def new_session() -> None:
    await stop_feed()
    if S.agent:
        await S.agent.stop()
    S.agent = AgentCore(S.provider)
    S.agent.start()
    S.segs.clear()
    S.speaker_order.clear()
    S.seq = 0
    S.version += 1
    S.error = ""


async def sign_out() -> None:
    if S.recording:
        await stop_recording()
    await stop_feed()
    if S.agent:
        await S.agent.stop()
        S.agent = None
    closed = set()
    for p in (S.provider, S.gemini_asr):
        if p is not None and id(p) not in closed:
            closed.add(id(p))
            try:
                await p.close()
            except Exception:
                pass
    S.provider = S.gemini_asr = None
    S.segs.clear()
    S.version += 1


async def activate(provider_name: str, key: str, model: str, gemini_key: str = "", gemini_model: str = "") -> None:
    if S.provider is not None:
        await sign_out()
    if provider_name == "anthropic":
        S.provider = AnthropicProvider(key, model)
        S.gemini_asr = GeminiProvider(gemini_key, gemini_model) if gemini_key else None
        S.gemini_asr_model = gemini_model
    else:
        S.provider = GeminiProvider(key, model)
        S.gemini_asr = S.provider
        S.gemini_asr_model = gemini_model or model
    S.source = "browser"
    await new_session()


async def _shutdown() -> None:
    await sign_out()


app.on_shutdown(_shutdown)


# =========================================================================== #
# Styles and client script (design ported from the 0xmoira prototype)
# =========================================================================== #
CSS = r"""
:root { --ink:#000; --paper:#fff; --rule:2px solid var(--ink);
  --mono: ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --sans: "Helvetica Neue",Helvetica,Arial,sans-serif; }
html.invert { --ink:#fff; --paper:#000; }
html, body { height:100%; margin:0; background:var(--paper); }
.nicegui-content { padding:0 !important; gap:0 !important; height:100vh; display:block !important; }
.q-page, .q-layout, .q-page-container { min-height:0 !important; }
.oxm, .oxm * { box-sizing:border-box; }
.oxm { background:var(--paper); color:var(--ink); font-family:var(--mono); font-size:13px; line-height:1.45;
  display:flex; flex-direction:column; height:100vh; overflow:hidden; overflow-wrap:anywhere; -webkit-font-smoothing:antialiased; }
:where(.oxm) :where(p, h1, h2, h3, h4, ul, dl, dd, q) { margin:0; padding:0; line-height:inherit; font-size:inherit; letter-spacing:inherit; }
.oxm ::selection { background:var(--ink); color:var(--paper); }
.oxm button, .oxm select, .oxm input, .oxm textarea { font:inherit; color:inherit; background:var(--paper); }
.hidden { display:none !important; }
.muted { opacity:.62; }
.empty { opacity:.5; text-transform:uppercase; font-size:11px; padding:10px 16px; }

.bar { display:flex; align-items:center; gap:24px; padding:10px 16px; border-bottom:var(--rule); flex:none; }
.bar h1 { font-family:var(--sans); font-weight:900; font-size:20px; line-height:.9; letter-spacing:-.04em; text-transform:uppercase; white-space:nowrap; }
.status { display:flex; align-items:center; gap:8px; text-transform:uppercase; min-width:0; flex:1 1 200px; }
.status span:last-child { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.dot { width:10px; height:10px; border:var(--rule); display:inline-block; flex:none; }
.dot.ok { background:var(--ink); }
.dot.err { background:repeating-linear-gradient(45deg,var(--ink) 0 2px,var(--paper) 2px 4px); }
.controls { margin-left:auto; display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }
.controls label { display:flex; gap:6px; align-items:center; text-transform:uppercase; font-size:11px; }
.oxm select { border:var(--rule); padding:4px 6px; appearance:none; -webkit-appearance:none; text-transform:uppercase; cursor:pointer; border-radius:0; }
.btn { border:var(--rule); padding:6px 12px; cursor:pointer; text-transform:uppercase; letter-spacing:.04em; font-weight:700; white-space:nowrap; border-radius:0; }
.btn:hover, .btn:focus-visible { background:var(--ink) !important; color:var(--paper) !important; outline:none; }
.btn:active { transform:translate(1px,1px); }
.btn[disabled] { opacity:.35; cursor:not-allowed; }
.btn-small { padding:2px 8px; font-size:11px; font-weight:400; }
.btn-rec.on { background:var(--ink); color:var(--paper); animation:oxblink 1s steps(2,start) infinite; }
@keyframes oxblink { to { visibility:hidden; } }
.chips { display:flex; gap:4px; align-items:center; }
.chips .spk { cursor:pointer; }
.chips .spk.sel { outline:3px solid var(--ink); outline-offset:2px; }

.foot { border-top:var(--rule); border-bottom:none; font-size:11px; text-transform:uppercase; gap:20px; flex-wrap:wrap; row-gap:4px; }
.foot span { white-space:nowrap; }
.level { width:120px; height:10px; border:var(--rule); display:inline-block; position:relative; flex:none; }
.level i { position:absolute; inset:0; width:0; background:var(--ink); }

.grid { flex:1 1 auto; display:grid; grid-template-columns:minmax(0,4fr) minmax(0,5fr) minmax(0,3fr); min-height:0; overflow:hidden; }
.panel { display:flex; flex-direction:column; min-height:0; min-width:0; overflow:hidden; }
.panel-transcript, .panel-overall { border-right:var(--rule); }
.panel-side { overflow-y:auto; }
.panel-head { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:6px 16px; border-bottom:var(--rule);
  font-family:var(--sans); font-weight:900; font-size:11px; letter-spacing:.12em; text-transform:uppercase; flex:none; }
.panel-head .muted { font-family:var(--mono); font-weight:400; letter-spacing:0; white-space:nowrap; }

.transcript { flex:1 1 auto; min-height:0; overflow-y:auto; padding:16px; font-size:15px; line-height:1.6; }
.transcript .empty { padding:0; }
.seg { margin-bottom:12px; display:grid; grid-template-columns:52px 26px 1fr; gap:10px; align-items:start; }
.seg time { font-size:11px; opacity:.62; padding-top:4px; white-space:nowrap; }
.seg.pending p { opacity:.6; }
.seg.error p::after { content:" [ANALYSIS FAILED]"; font-size:11px; opacity:.62; }
.seg .ev { font-size:11px; line-height:1.4; opacity:.75; margin-top:2px; }
.seg .ev b { font-family:var(--sans); font-weight:900; }
.interim { opacity:.5; font-style:italic; padding:4px 16px 8px 104px; min-height:0; flex:none; font-size:14px; }
.interim:empty { display:none; }

.spk { white-space:nowrap; overflow-wrap:normal; line-height:1; display:inline-flex; align-items:center; justify-content:center; width:24px; height:22px; margin-top:2px;
  font-family:var(--sans); font-weight:900; font-size:10px; border:var(--rule); flex:none; cursor:help; }
.spk-0 { background:var(--ink); color:var(--paper); }
.spk-1 { background:var(--paper); color:var(--ink); }
.spk-2 { background:repeating-linear-gradient(45deg,var(--ink) 0 2px,var(--paper) 2px 6px); color:var(--ink); text-shadow:0 0 3px var(--paper),0 0 3px var(--paper); }
.spk-3 { background:var(--paper); color:var(--ink); border-style:dashed; }
.spk-4 { background:radial-gradient(var(--ink) 1px,var(--paper) 1.5px) 0 0/5px 5px; color:var(--ink); text-shadow:0 0 3px var(--paper),0 0 3px var(--paper); }
.spk-5 { background:var(--ink); color:var(--paper); border-style:double; border-width:4px; }

mark { background:var(--ink); color:var(--paper); padding:0 3px; cursor:help; }
mark.sev-low { background:transparent; color:var(--ink); border-bottom:var(--rule); padding:0; }
mark.sev-medium { outline:2px solid var(--ink); background:transparent; color:var(--ink); }
sup.marker { font-size:9px; font-weight:700; margin-left:2px; font-family:var(--sans); }

.feed { border-top:var(--rule); display:flex; flex-direction:column; flex:none; max-height:45%; }
.feed textarea { border:none; border-bottom:var(--rule); padding:10px 16px; resize:vertical; width:100%; outline:none; min-height:90px; font-size:13px; }
.feed-controls { display:flex; gap:16px; align-items:center; padding:8px 16px; flex-wrap:wrap; }
.feed-controls label { display:flex; gap:8px; align-items:center; font-size:11px; text-transform:uppercase; }
.oxm input[type=range] { accent-color:var(--ink); width:140px; }

.block { border-bottom:var(--rule); flex:none; }
.bars { padding:4px 16px 8px; display:grid; gap:5px; }
.bar-row { display:grid; grid-template-columns:minmax(80px,112px) 1fr 34px; align-items:center; gap:10px; font-size:10px; text-transform:uppercase; }
.bar-track { height:10px; border:var(--rule); position:relative; overflow:hidden; }
.bar-fill { position:absolute; top:0; bottom:0; left:0; width:0; background:var(--ink); transition:width .5s steps(20), left .5s steps(20); }
.bar-track.bipolar::after { content:""; position:absolute; left:calc(50% - 1px); top:0; bottom:0; width:2px; background:var(--ink); }
.bar-val { text-align:right; font-variant-numeric:tabular-nums; }
h4.sub { font-family:var(--mono); font-weight:400; font-size:9px; letter-spacing:.14em; opacity:.62; padding:8px 16px 0; text-transform:uppercase; }
.mood-line { padding:8px 16px 2px; font-family:var(--sans); font-weight:900; font-size:13px; text-transform:uppercase; letter-spacing:-.01em; }
.mood-line .sep { opacity:.4; }
.intent { padding:2px 16px 4px; font-size:12px; font-style:italic; }

.action { padding:12px 16px 14px; }
.action .what { font-family:var(--sans); font-weight:900; font-size:19px; line-height:1.08; letter-spacing:-.025em; text-transform:uppercase; margin:8px 0 6px; }
.action .meta { display:flex; gap:8px; align-items:center; font-size:10px; text-transform:uppercase; letter-spacing:.08em; }
.tag { border:var(--rule); padding:1px 6px; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; white-space:nowrap; }
.tag.inv { background:var(--ink); color:var(--paper); }
.action .intent { padding:0; }
.prev-actions { list-style:none; padding:0 16px 10px; }
.prev-actions li { font-size:11px; opacity:.7; padding:3px 0; border-top:1px solid var(--ink); display:grid; grid-template-columns:26px 1fr; gap:8px; }

.tension { display:grid; grid-template-columns:74px 1fr; gap:12px; padding:10px 16px 6px; align-items:end; }
.tension .num { font-family:var(--sans); font-weight:900; font-size:56px; line-height:.85; letter-spacing:-.05em; }
.tension .num small { font-size:14px; letter-spacing:0; opacity:.62; }
.tension svg { width:100%; height:84px; border:var(--rule); display:block; color:var(--ink); }
.legend { display:flex; gap:10px; flex-wrap:wrap; padding:2px 16px 10px; font-size:10px; text-transform:uppercase; align-items:center; }
.legend span.ln { display:inline-block; width:22px; height:0; border-top:2px solid var(--ink); vertical-align:middle; margin-right:4px; }

.speakers, .ideas, .signals { list-style:none; }
.speaker { display:grid; grid-template-columns:26px 1fr; gap:10px; padding:8px 16px; border-bottom:1px solid var(--ink); }
.speaker:last-child, .idea:last-child, .signal:last-child { border-bottom:none; }
.speaker .name { font-family:var(--sans); font-weight:900; text-transform:uppercase; font-size:12px; }
.speaker .stats { font-size:10px; text-transform:uppercase; opacity:.7; margin-top:2px; }
.share { height:6px; border:1px solid var(--ink); margin-top:4px; position:relative; }
.share i { position:absolute; top:0; bottom:0; left:0; background:var(--ink); }
.idea { display:grid; grid-template-columns:44px 1fr; gap:10px; padding:8px 16px; border-bottom:1px solid var(--ink); }
.novelty { font-family:var(--sans); font-weight:900; font-size:18px; line-height:1; }
.novelty small { display:block; font-family:var(--mono); font-weight:400; font-size:9px; letter-spacing:.1em; opacity:.62; }
.idea .why, .idea .spin { font-size:11px; opacity:.75; margin-top:2px; }
.signal { padding:6px 16px; border-bottom:1px solid var(--ink); font-size:12px; display:grid; grid-template-columns:26px 1fr; gap:10px; }
.signal .cat { font-size:10px; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }

.reliability { display:grid; grid-template-columns:70px 1fr 40px; gap:10px; align-items:center; padding:8px 16px; border-bottom:var(--rule); font-size:10px; text-transform:uppercase; flex:none; }
.ov-scroll { flex:1 1 auto; min-height:0; overflow-y:auto; display:flex; flex-direction:column; }
.overall { flex:1 1 auto; padding:20px 24px 28px; }
.overall.busy { opacity:.55; }
.ov-headline { font-family:var(--sans); font-weight:900; font-size:30px; line-height:1.02; letter-spacing:-.035em; text-transform:uppercase; margin-bottom:14px !important; overflow-wrap:break-word; }
.ov-lead { font-size:15px; line-height:1.55; margin-bottom:20px !important; }
.ov-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); border:var(--rule); margin-bottom:20px; }
.ov-card { padding:10px 12px 12px; border-right:var(--rule); border-bottom:var(--rule); }
.ov-card:nth-child(2n) { border-right:none; }
.ov-card:nth-last-child(-n+2):nth-child(odd), .ov-card:last-child { border-bottom:none; }
.ov-card.wide { grid-column:1 / -1; border-right:none; background:var(--ink); color:var(--paper); }
.ov-card h3, .ov-sub { font-family:var(--sans); font-weight:900; font-size:10px; letter-spacing:.14em; text-transform:uppercase; margin-bottom:4px !important; }
.ov-card p { font-size:13px; line-height:1.5; }
.ov-card.wide p { font-family:var(--sans); font-weight:900; font-size:16px; line-height:1.2; text-transform:uppercase; letter-spacing:-.01em; }
.ov-sub { margin:0 0 8px !important; }
.tells { list-style:none; border:var(--rule); margin-bottom:20px; }
.tells li { display:grid; grid-template-columns:26px 1fr; gap:10px; padding:6px 10px; border-bottom:1px solid var(--ink); font-size:13px; }
.tells li:last-child { border-bottom:none; }
.tells q { font-style:italic; }
.tells .pat { font-size:11px; opacity:.75; }
.ov-speakers { display:grid; gap:16px; margin-bottom:18px; }
.ov-spk { display:grid; grid-template-columns:26px 1fr; gap:12px; }
.ov-spk .role { font-family:var(--sans); font-weight:900; text-transform:uppercase; font-size:13px; margin-bottom:4px; }
.ov-spk dl { display:grid; grid-template-columns:96px 1fr; gap:3px 10px; font-size:12.5px; line-height:1.45; }
.ov-spk dt { font-size:9px; letter-spacing:.12em; text-transform:uppercase; opacity:.62; padding-top:3px; }
.ov-spk dd.bottom { font-weight:700; }
.ov-trend { border-top:var(--rule); padding-top:10px; font-style:italic; font-size:13px; }

/* sign-in */
.login { display:flex; align-items:center; justify-content:center; flex:1 1 auto; padding:24px; overflow:auto; }
.login-box { width:min(560px,100%); border:var(--rule); }
.login-title { font-family:var(--sans); font-weight:900; font-size:64px; line-height:.86; letter-spacing:-.05em; text-transform:uppercase; padding:20px 20px 16px; border-bottom:var(--rule); }
.login-title small { display:block; font-family:var(--mono); font-weight:400; font-size:11px; letter-spacing:.12em; margin-top:12px; opacity:.7; }
.login-body { padding:16px 20px 20px; display:grid; gap:14px; }
.seg-ctl { display:grid; grid-template-columns:1fr 1fr; border:var(--rule); }
.seg-ctl button { border:none; padding:10px; cursor:pointer; font-weight:700; text-transform:uppercase; letter-spacing:.06em; border-radius:0; }
.seg-ctl button + button { border-left:var(--rule); }
.seg-ctl button.sel { background:var(--ink); color:var(--paper); }
.field { display:grid; gap:4px; }
.field label { font-size:10px; letter-spacing:.12em; text-transform:uppercase; opacity:.7; }
.field input, .field select { border:var(--rule); padding:8px 10px; width:100%; outline:none; border-radius:0; }
.field input:focus { outline:2px solid var(--ink); outline-offset:2px; }
.hint { font-size:11px; opacity:.7; }
.msg { border:var(--rule); padding:8px 10px; font-size:12px; }
.msg.err { background:repeating-linear-gradient(45deg,transparent 0 6px,rgba(127,127,127,.15) 6px 12px); font-weight:700; }
.row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.check { display:flex; gap:8px; align-items:center; font-size:11px; text-transform:uppercase; }
.check input { accent-color:var(--ink); width:16px; height:16px; }

@media (max-width:1100px) {
  .oxm { height:auto; min-height:100vh; overflow:auto; }
  .grid { grid-template-columns:minmax(0,1fr) minmax(0,1fr); overflow:visible; }
  .panel-overall { border-right:none; }
  .panel-side { grid-column:1 / -1; border-top:var(--rule); }
  .transcript, .ov-scroll { max-height:70vh; }
}
@media (max-width:720px) {
  .grid { grid-template-columns:minmax(0,1fr); }
  .panel-transcript, .panel-overall { border-right:none; border-bottom:var(--rule); }
  .ov-grid { grid-template-columns:minmax(0,1fr); }
  .bar { flex-wrap:wrap; gap:12px; }
  .controls { margin-left:0; }
}
"""

SAMPLE = """Maya: Okay, quick one. Where are we on the Friday release?
Jonas: Honestly, it's fine. A few tests are flaky, but that's not a big deal.
Maya: Flaky how? Last sprint "flaky" meant two days of hotfixes.
Jonas: That was different. And frankly, some people could have reviewed my PR earlier.
Priya: I reviewed it within an hour of you asking. I only asked for numbers on the load test.
Jonas: Fine. Whatever. We'll ship it Friday like every time.
Maya: What if we ship to five percent of users first and only roll forward if the error rate stays flat?
Priya: Das wäre gut, aber wer übernimmt am Wochenende das Monitoring?
Jonas: Not me. I've carried this release alone for three weeks.
Maya: Then let's decide now: who owns monitoring, and which error rate stops the rollout?"""

JS = r"""
window.oxm = window.oxm || {};
(function () {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  oxm.speech = {
    rec: null, active: false, lang: 'en-US',
    start(lang) {
      if (!SR) { emitEvent('oxm_speech_error', {error: 'not-supported'}); return; }
      this.lang = lang; this.active = true; this._start();
    },
    _start() {
      const r = new SR();
      r.lang = this.lang; r.continuous = true; r.interimResults = true; r.maxAlternatives = 1;
      r.onresult = (ev) => {
        let interim = '';
        for (let i = ev.resultIndex; i < ev.results.length; i++) {
          const res = ev.results[i], t = res[0].transcript;
          if (res.isFinal) { if (t.trim()) emitEvent('oxm_final', {text: t.trim(), confidence: res[0].confidence || 0}); }
          else interim += t;
        }
        const el = document.getElementById('oxm-interim');
        if (el) el.textContent = interim;
      };
      r.onerror = (ev) => {
        if (ev.error === 'no-speech' || ev.error === 'aborted') return;
        if (ev.error === 'not-allowed' || ev.error === 'service-not-allowed') this.active = false;
        emitEvent('oxm_speech_error', {error: ev.error});
      };
      r.onend = () => { if (this.active) setTimeout(() => { if (this.active) this._start(); }, 150); };
      this.rec = r;
      try { r.start(); } catch (err) {}
    },
    stop() {
      this.active = false;
      if (this.rec) { try { this.rec.stop(); } catch (err) {} }
      const el = document.getElementById('oxm-interim'); if (el) el.textContent = '';
    },
  };
  oxm.stick = true;
  oxm.init = function () {
    const sc = document.querySelector('.transcript');
    if (!sc || sc.dataset.oxm) return;
    sc.dataset.oxm = '1';
    sc.addEventListener('scroll', () => { oxm.stick = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 80; });
    new MutationObserver(() => { if (oxm.stick) sc.scrollTop = sc.scrollHeight; }).observe(sc, {childList: true, subtree: true});
  };
  oxm.apply = function (s) {
    const $ = (id) => document.getElementById(id);
    if (!$('oxm-status')) return;
    $('oxm-dot').className = 'dot ' + (s.err ? 'err' : (s.live ? 'ok' : ''));
    $('oxm-status').textContent = s.status;
    $('oxm-status').title = s.status;
    const rec = $('oxm-rec');
    rec.classList.toggle('on', s.recording);
    rec.textContent = s.recording ? '■ STOP' : '● REC';
    rec.disabled = s.source === 'text';
    $('oxm-source').value = s.source;
    $('oxm-lang').value = s.lang;
    $('oxm-chips').classList.toggle('hidden', s.source !== 'browser' && s.source !== 'text');
    document.querySelectorAll('#oxm-chips .spk').forEach((c) => c.classList.toggle('sel', c.dataset.spk === s.speaker));
    $('oxm-feed').classList.toggle('hidden', s.source !== 'text');
    $('oxm-stream').textContent = s.streaming ? '■ STOP' : '▶ STREAM';
    $('f-model').textContent = s.model;
    $('f-latency').textContent = s.latency;
    $('f-queue').textContent = s.queue;
    $('f-count').textContent = s.count;
    $('f-spend').textContent = s.spend;
    $('f-level').style.width = s.level + '%';
  };
  oxm.cmd = function (cmd, value) { emitEvent('oxm_cmd', {cmd: cmd, value: value === undefined ? null : value}); };
  oxm.stream = function () {
    emitEvent('oxm_cmd', {cmd: 'stream', value: {text: document.getElementById('oxm-feed-text').value,
      wpm: +document.getElementById('oxm-wpm').value}});
  };
  oxm.sample = function (t) { document.getElementById('oxm-feed-text').value = t; };
  oxm.invert = function () {
    const on = document.documentElement.classList.toggle('invert');
    try { localStorage.setItem('oxm-invert', on ? '1' : '0'); } catch (err) {}
  };
  try { if (localStorage.getItem('oxm-invert') === '1') document.documentElement.classList.add('invert'); } catch (err) {}

  // ---- sign-in ----
  oxm.login = {
    provider: 'anthropic',
    pick(p) {
      this.provider = p;
      document.querySelectorAll('#lg-prov button').forEach((b) => b.classList.toggle('sel', b.dataset.p === p));
      document.getElementById('lg-key-label').textContent = p === 'anthropic' ? 'ANTHROPIC API KEY' : 'GEMINI API KEY';
      document.getElementById('lg-key').placeholder = p === 'anthropic' ? 'sk-ant-…' : 'AIza…';
      document.getElementById('lg-step2').classList.add('hidden');
      document.getElementById('lg-asr').classList.toggle('hidden', p !== 'anthropic');
      this.msg('');
    },
    verify(useEnv) {
      const key = document.getElementById('lg-key').value.trim();
      if (!key && !useEnv) { this.msg('Enter an API key first.', true); return; }
      this.msg('VERIFYING…');
      emitEvent('oxm_login', {action: 'verify', provider: this.provider, key: key, use_env: !!useEnv});
    },
    models(list, def) {
      const sel = document.getElementById('lg-model');
      sel.innerHTML = '';
      list.forEach((m) => { const o = document.createElement('option'); o.value = m; o.textContent = m; if (m === def) o.selected = true; sel.appendChild(o); });
      document.getElementById('lg-step2').classList.remove('hidden');
      this.msg('KEY VERIFIED. ' + list.length + ' MODELS AVAILABLE.');
    },
    enter() {
      this.msg('SIGNING IN…');
      emitEvent('oxm_login', {action: 'enter', model: document.getElementById('lg-model').value,
        remember: document.getElementById('lg-remember').checked,
        gemini_key: (document.getElementById('lg-gkey') || {value: ''}).value.trim()});
    },
    msg(text, err) {
      const m = document.getElementById('lg-msg');
      m.textContent = text; m.className = 'msg' + (err ? ' err' : '') + (text ? '' : ' hidden');
    },
  };
})();
"""


def head() -> None:
    ui.add_head_html(f"<style>{CSS}</style><script>{JS}</script>")


def event_args(ev: Any) -> Dict[str, Any]:
    a = ev.args
    if isinstance(a, list) and len(a) == 1:
        a = a[0]
    return a if isinstance(a, dict) else {}


# =========================================================================== #
# Sign-in page
# =========================================================================== #
@ui.page("/login")
def login_page() -> None:
    head()
    ui.page_title("0xmoira · sign in")
    client_id = ui.context.client.id
    saved = load_credentials()
    env_a, env_g = bool(os.getenv("ANTHROPIC_API_KEY")), bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    saved_html = ""
    if saved.get("provider") and saved.get("keys", {}).get(saved["provider"]):
        k = saved["keys"][saved["provider"]]
        saved_html = (f'<div class="msg"><b>SAVED ON THIS MAC:</b> {e(saved["provider"].upper())} · {e(saved.get("model", ""))} · {e(mask_key(k))}'
                      f'<div class="row" style="margin-top:8px"><button class="btn" onclick="emitEvent(\'oxm_login\',{{action:\'saved\'}})">CONTINUE</button>'
                      f'<button class="btn btn-small" onclick="emitEvent(\'oxm_login\',{{action:\'forget\'}})">FORGET</button></div></div>')
    env_btn = ('<button class="btn btn-small" id="lg-env" onclick="oxm.login.verify(true)">USE KEY FROM ENVIRONMENT</button>'
               if env_a or env_g else "")

    ui.html(f"""
<div class="oxm"><div class="login"><div class="login-box">
  <div class="login-title">0X<br>MOIRA<small>REAL-TIME PSYCHOLOGICAL MEETING COPILOT · SIGN IN</small></div>
  <div class="login-body">
    {saved_html}
    <div class="seg-ctl" id="lg-prov">
      <button class="sel" data-p="anthropic" onclick="oxm.login.pick('anthropic')">ANTHROPIC CLAUDE</button>
      <button data-p="gemini" onclick="oxm.login.pick('gemini')">GOOGLE GEMINI</button>
    </div>
    <div class="field">
      <label id="lg-key-label" for="lg-key">ANTHROPIC API KEY</label>
      <input id="lg-key" type="password" autocomplete="off" spellcheck="false" placeholder="sk-ant-…"
             onkeydown="if(event.key==='Enter')oxm.login.verify(false)">
      <div class="hint">The key is verified against the provider, then kept in memory on this Mac only. It is never sent back to the browser.</div>
    </div>
    <div class="row"><button class="btn" onclick="oxm.login.verify(false)">VERIFY KEY</button>{env_btn}</div>
    <div id="lg-msg" class="msg hidden"></div>
    <div id="lg-step2" class="hidden" style="display:grid;gap:14px">
      <div class="field"><label for="lg-model">ANALYSIS MODEL</label><select id="lg-model"></select></div>
      <div class="field" id="lg-asr">
        <label for="lg-gkey">GEMINI API KEY · OPTIONAL</label>
        <input id="lg-gkey" type="password" autocomplete="off" spellcheck="false" placeholder="AIza… (enables server-mic transcription with speaker labels)">
      </div>
      <label class="check"><input type="checkbox" id="lg-remember" checked> REMEMBER ON THIS MAC (~/.oxmoira, owner-only)</label>
      <div class="row"><button class="btn" onclick="oxm.login.enter()">ENTER ▸</button></div>
    </div>
  </div>
</div></div></div>""", sanitize=False)

    async def on_login(ev: Any) -> None:
        a = event_args(ev)
        action = a.get("action")
        try:
            if action == "forget":
                clear_credentials()
                ui.navigate.reload()
                return
            if action == "saved":
                prov = saved["provider"]
                keys = saved.get("keys", {})
                await activate(prov, keys[prov], saved.get("model", ""), keys.get("gemini", "") if prov == "anthropic" else "",
                               saved.get("gemini_model", ""))
                ui.navigate.to("/")
                return
            if action == "verify":
                prov = a.get("provider", "anthropic")
                key = (a.get("key") or "").strip()
                if a.get("use_env"):
                    key = os.getenv("ANTHROPIC_API_KEY", "") if prov == "anthropic" else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "")
                    if not key:
                        raise LoginError(f"No {prov} key in the environment.")
                ids = await (verify_anthropic(key) if prov == "anthropic" else verify_gemini(key))
                default = default_anthropic_model(ids) if prov == "anthropic" else default_gemini_model(ids)
                PENDING[client_id] = {"provider": prov, "key": key, "models": ids}
                ui.run_javascript(f"oxm.login.models({json.dumps(ids)}, {json.dumps(default)})")
                return
            if action == "enter":
                p = PENDING.get(client_id)
                if not p:
                    raise LoginError("Verify a key first.")
                model = a.get("model") or p["models"][0]
                if model not in p["models"]:
                    raise LoginError("Pick a model from the list.")
                gkey, gmodel = "", ""
                if p["provider"] == "anthropic" and a.get("gemini_key"):
                    gkey = a["gemini_key"]
                    gmodel = default_gemini_model(await verify_gemini(gkey))
                elif p["provider"] == "gemini":
                    gmodel = default_gemini_model(p["models"])
                await activate(p["provider"], p["key"], model, gkey, gmodel)
                if a.get("remember"):
                    keys = {p["provider"]: p["key"]}
                    if gkey:
                        keys["gemini"] = gkey
                    save_credentials({"provider": p["provider"], "model": model, "gemini_model": gmodel, "keys": keys})
                PENDING.pop(client_id, None)
                ui.navigate.to("/")
        except LoginError as exc:
            ui.run_javascript(f"oxm.login.msg({json.dumps(str(exc))}, true)")
        except Exception as exc:
            log.exception("sign-in failed")
            ui.run_javascript(f"oxm.login.msg({json.dumps(f'{type(exc).__name__}: {exc}')}, true)")

    ui.on("oxm_login", on_login)


# =========================================================================== #
# Renderers (all dynamic text is escaped)
# =========================================================================== #
def results_by_uid() -> Dict[str, AnalysisResult]:
    return {r.utterance_id: r for r in (S.agent.history if S.agent else [])}


def highlight(text: str, flags: List[Any]) -> str:
    spans: List[Tuple[int, int, Any]] = []
    low = text.lower()
    for f in flags:
        i = low.find(f.phrase.lower().strip())
        if i < 0:
            continue
        j = i + len(f.phrase.strip())
        if all(j <= a or i >= b for a, b, _ in spans):
            spans.append((i, j, f))
    spans.sort(key=lambda x: x[0])
    out, pos = [], 0
    for i, j, f in spans:
        out.append(e(text[pos:i]))
        out.append(f'<mark class="sev-{e(f.severity)}" title="{e(f.category.upper())}: {e(f.reason)}">{e(text[i:j])}</mark>'
                   f'<sup class="marker">{e(f.category[:3].upper())}</sup>')
        pos = j
    out.append(e(text[pos:]))
    return "".join(out)


def render_transcript() -> str:
    if not S.segs:
        return ('<p class="empty">PRESS ● REC AND SPEAK, OR SWITCH THE SOURCE TO TEXT AND STREAM A TRANSCRIPT. '
                'IN BROWSER-MIC MODE, CLICK A SPEAKER BADGE IN THE HEADER WHEN THE SPEAKER CHANGES.</p>')
    res = results_by_uid()
    rows = []
    for s in S.segs[-300:]:
        r = res.get(s.uid)
        cls, body, ev = "seg", e(s.text), ""
        if r is None:
            cls += " pending"
        elif r.insight is not None:
            ins = r.insight
            body = highlight(s.text, ins.flags)
            ev = f'<div class="ev"><b>T{ins.cognitive_tension_score}</b> · {e(ins.mood.upper())} · {e(ins.evidence)}</div>'
        elif r.skipped_reason.startswith(("error", "parse-failed")):
            cls += " error"
            ev = f'<div class="ev">{e(r.skipped_reason[:160])}</div>'
        m, sec = divmod(int(s.t), 60)
        rows.append(f'<div class="{cls}"><time>{m:02d}:{sec:02d}</time>{badge(s.speaker)}<div><p>{body}</p>{ev}</div></div>')
    return "".join(rows)


def render_overall() -> str:
    a = S.agent
    syn = a.synthesis if a else None
    if a is None:
        meta = "—"
    elif a.synth_running:
        meta = "UPDATING…"
    elif a.synth_error:
        meta = "FAILED · " + a.synth_error[:60]
    elif syn:
        meta = f"{time.strftime('%H:%M', time.localtime(syn.ts))} · {syn.utterances_covered} LINES · ${syn.cost_usd:.3f}"
    else:
        meta = "WAITING FOR MATERIAL"
    head_html = (f'<div class="panel-head"><span>OVERALL ANALYSIS</span><span class="muted">{e(meta)}</span>'
                 f'<button class="btn btn-small" onclick="oxm.cmd(\'analyse\')">ANALYSE NOW</button></div>')
    rel = syn.reliability if syn else 0
    rel_html = (f'<div class="reliability"><span>EVIDENCE</span><div class="bar-track"><div class="bar-fill" style="width:{rel}%"></div></div>'
                f'<span class="bar-val">{rel if syn else "—"}</span></div>')
    if syn is None:
        every = f"EVERY {SYNTH_EVERY} ANALYSED UTTERANCES" if SYNTH_EVERY else "ON DEMAND"
        body = (f'<p class="empty" style="padding:0">THE OVERALL ANALYSIS IS BUILT FROM EVERYTHING SAID AND REFRESHES {every}. '
                'IT PROFILES EACH SPEAKER, THE DYNAMICS BETWEEN THEM AND WHAT NOBODY SAYS.</p>')
    else:
        cards = (f'<div class="ov-card wide"><h3>NEXT MOVE</h3><p>{e(syn.recommendation)}</p></div>'
                 f'<div class="ov-card"><h3>DYNAMICS</h3><p>{e(syn.dynamics)}</p></div>'
                 f'<div class="ov-card"><h3>UNDERCURRENT</h3><p>{e(syn.undercurrent or "—")}</p></div>'
                 f'<div class="ov-card"><h3>RISKS</h3><p>{e(syn.risks or "—")}</p></div>'
                 f'<div class="ov-card"><h3>TREND</h3><p>{e(syn.trend or "—")}</p></div>')
        tells = "".join(f'<li>{badge(t.who)}<div><q>{e(t.quote)}</q><div class="pat">{e(t.pattern)}</div></div></li>'
                        for t in syn.tells)
        tells_html = f'<h3 class="ov-sub">TELLS</h3><ul class="tells">{tells}</ul>' if tells else ""
        spk = []
        for p in syn.speakers:
            fields = [("EVIDENCE", p.evidence), ("STATE", p.state), ("MOTIVES", p.motives), ("DEFENCES", p.defenses),
                      ("RELATION", p.relation), ("SUBTEXT", p.subtext), ("LEVER", p.lever), ("CREDIBILITY", p.credibility)]
            dl = "".join(f"<dt>{k}</dt><dd>{e(v)}</dd>" for k, v in fields if v)
            if p.bottom_line:
                dl += f'<dt>BOTTOM LINE</dt><dd class="bottom">{e(p.bottom_line)}</dd>'
            spk.append(f'<div class="ov-spk">{badge(p.label)}<div><div class="role">{e(p.label)} · {e(p.role)}</div><dl>{dl}</dl></div></div>')
        body = (f'<h2 class="ov-headline">{e(syn.headline)}</h2><p class="ov-lead">{e(syn.situation)}</p>'
                f'<div class="ov-grid">{cards}</div>{tells_html}'
                f'<h3 class="ov-sub">SPEAKERS</h3><div class="ov-speakers">{"".join(spk)}</div>')
    busy = " busy" if a and a.synth_running else ""
    return f'{head_html}{rel_html}<div class="ov-scroll"><div class="overall{busy}">{body}</div></div>'


def bar(label: str, v: Optional[float], bipolar: bool = False) -> str:
    if v is None:
        return (f'<div class="bar-row"><span>{e(label)}</span><div class="bar-track{" bipolar" if bipolar else ""}"></div>'
                f'<span class="bar-val">—</span></div>')
    if bipolar:
        w = abs(v) / 2
        left = 50 if v >= 0 else 50 - w
        fill = f'<div class="bar-fill" style="left:{left:.1f}%;width:{w:.1f}%"></div>'
        return (f'<div class="bar-row"><span>{e(label)}</span><div class="bar-track bipolar">{fill}</div>'
                f'<span class="bar-val">{v:+.0f}</span></div>')
    return (f'<div class="bar-row"><span>{e(label)}</span><div class="bar-track"><div class="bar-fill" style="width:{max(0, min(100, v)):.0f}%"></div></div>'
            f'<span class="bar-val">{v:.0f}</span></div>')


DASH = ["", "6 3", "2 3", "10 3 2 3", "1 2", "12 4"]
INTERVENTION_LABEL = {"probe_question": "PROBE", "reframe": "REFRAME", "acknowledge": "ACKNOWLEDGE", "pause": "PAUSE",
                      "boundary": "BOUNDARY", "redirect": "REDIRECT", "hold": "HOLD"}


def tension_svg(results: List[AnalysisResult]) -> Tuple[str, str]:
    pts = [(r.speaker_id, r.insight.cognitive_tension_score) for r in results if r.insight][-40:]
    if not pts:
        return '<div class="empty" style="padding:30px 0;border:var(--rule)">NO DATA YET</div>', ""
    w, h = 300, 84
    n = len(pts)

    def x(i: int) -> float:
        return 6 + (w - 12) * (i / max(1, n - 1))

    def y(t: float) -> float:
        return h - 6 - (h - 12) * (t - 1) / 9

    parts = [f'<line x1="0" x2="{w}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="currentColor" stroke-width="1" opacity=".25" stroke-dasharray="2 4"/>'
             for t in (4, 7)]
    legend = []
    for label in dict.fromkeys(s for s, _ in pts):
        idx = spk_index(label)
        coords = [(x(i), y(t)) for i, (s, t) in enumerate(pts) if s == label]
        if len(coords) > 1:
            pl = " ".join(f"{a:.1f},{b:.1f}" for a, b in coords)
            parts.append(f'<polyline points="{pl}" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="{DASH[idx]}"/>')
        fill = "currentColor" if idx % 2 == 0 else "var(--paper)"
        parts += [f'<rect x="{a - 3:.1f}" y="{b - 3:.1f}" width="6" height="6" style="fill:{fill}" stroke="currentColor" stroke-width="1.5"/>'
                  for a, b in coords]
        legend.append(f'<span><span class="ln" style="border-top-style:{"solid" if not DASH[idx] else "dashed"}"></span>{e(label)}</span>')
    svg = f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">{"".join(parts)}</svg>'
    return svg, f'<div class="legend">{"".join(legend)}</div>'


def render_side() -> str:
    a = S.agent
    hist = a.history if a else []
    analysed = [r for r in hist if r.insight]
    last = analysed[-1] if analysed else None

    # intervention
    if last:
        ins = last.insight
        typ = INTERVENTION_LABEL.get(ins.intervention_type, ins.intervention_type.upper())
        prev = "".join(f'<li>{badge(r.speaker_id)}<span>{e(r.insight.tactical_action)}</span></li>'
                       for r in reversed(analysed[-4:-1]) if r.insight.intervention_type != "hold")
        action = (f'<div class="action"><div class="meta">{badge(last.speaker_id)}<span class="tag inv">{e(typ)}</span>'
                  f'<span class="tag">TENSION {ins.cognitive_tension_score}/10</span></div>'
                  f'<p class="what">{e(ins.tactical_action)}</p>'
                  f'<p class="intent">{e(ins.underlying_intent)}</p></div>'
                  + (f'<ul class="prev-actions">{prev}</ul>' if prev else ""))
    else:
        action = '<p class="empty">THE NEXT MOVE FOR YOU APPEARS HERE AFTER EACH ANALYSED UTTERANCE.</p>'
    blocks = [f'<div class="block"><div class="panel-head"><span>YOUR NEXT MOVE</span>'
              f'<span class="muted">{e(f"{last.latency_s:.1f} S" if last else "—")}</span></div>{action}</div>']

    # tension
    svg, legend = tension_svg(hist)
    num = f'{last.insight.cognitive_tension_score}<small>/10</small>' if last else '—'
    blocks.append(f'<div class="block"><div class="panel-head"><span>TENSION</span><span class="muted">LAST 40 UTTERANCES</span></div>'
                  f'<div class="tension"><div class="num">{num}</div>{svg}</div>{legend}</div>')

    # indicators (smoothed, for the most recent analysed speaker)
    if last:
        sm = last.smoothed or {}
        ins = last.insight
        ind = (f'<p class="mood-line">{e(ins.mood)}<span class="sep"> · </span>{e(last.speaker_id)}'
               f'<span class="sep"> · </span>{e(INTERVENTION_LABEL.get(ins.intervention_type, ""))}</p>'
               f'<h4 class="sub">AFFECT</h4><div class="bars">{bar("VALENCE", sm.get("valence"), True)}{bar("AROUSAL", sm.get("arousal"))}</div>'
               f'<h4 class="sub">STATE</h4><div class="bars">{bar("STRESS", sm.get("stress"))}{bar("AGITATION", sm.get("agitation"))}'
               f'{bar("DEFENSIVENESS", sm.get("defensiveness"))}</div>'
               f'<h4 class="sub">CONDUCT</h4><div class="bars">{bar("DOMINANCE", sm.get("dominance"))}{bar("CONFIDENCE", sm.get("confidence"))}'
               f'{bar("COHERENCE", sm.get("coherence"))}</div>')
    else:
        ind = '<p class="empty">NO READINGS YET.</p>'
    blocks.append(f'<div class="block"><div class="panel-head"><span>INDICATORS</span><span class="muted">SMOOTHED · '
                  f'{e(last.speaker_id.upper()) if last else "—"}</span></div>{ind}</div>')

    # speakers
    stats = a.speaker_stats if a else {}
    total = sum(st.words for st in stats.values()) or 1
    spk_rows = []
    for label, st in stats.items():
        share = 100 * st.words / total
        t = st.ema.get("cognitive_tension_score")
        d = st.ema.get("dominance")
        spk_rows.append(f'<li class="speaker">{badge(label)}<div><div class="name">{e(label)}</div>'
                        f'<div class="stats">{st.words} WORDS · {share:.0f}% OF TALK · TENSION {f"{t:.1f}" if t is not None else "—"}'
                        f' · DOMINANCE {f"{d:.0f}" if d is not None else "—"}</div>'
                        f'<div class="share"><i style="width:{share:.0f}%"></i></div></div></li>')
    spk_html = f'<ul class="speakers">{"".join(spk_rows)}</ul>' if spk_rows else '<p class="empty">NO SPEAKERS YET.</p>'
    blocks.append(f'<div class="block"><div class="panel-head"><span>SPEAKERS</span><span class="muted">{len(stats)}</span></div>{spk_html}</div>')

    # signals
    flags = [(r.speaker_id, f) for r in analysed for f in r.insight.flags][-6:]
    sig = "".join(f'<li class="signal">{badge(s)}<div><div class="cat">{e(f.category)} · {e(f.severity)}</div>'
                  f'<q>{e(f.phrase)}</q><div class="muted">{e(f.reason)}</div></div></li>' for s, f in reversed(flags))
    blocks.append(f'<div class="block"><div class="panel-head"><span>SIGNALS</span><span class="muted">{len(flags)}</span></div>'
                  + (f'<ul class="signals">{sig}</ul>' if sig else '<p class="empty">NOTHING FLAGGED.</p>') + '</div>')

    # ideas
    ideas = [(r.speaker_id, i) for r in analysed for i in r.insight.ideas]
    ideas.sort(key=lambda x: -x[1].novelty)
    idea_rows = "".join(f'<li class="idea"><div class="novelty">{i.novelty}<small>NOVELTY</small></div><div><div>{e(i.idea)}</div>'
                        f'<div class="why">{e(i.why)} · {e(s)}</div><div class="spin">↳ {e(i.spin)}</div></div></li>'
                        for s, i in ideas[:6])
    blocks.append(f'<div class="block"><div class="panel-head"><span>IDEAS</span><span class="muted">{len(ideas)}</span></div>'
                  + (f'<ul class="ideas">{idea_rows}</ul>' if idea_rows else '<p class="empty">NO REAL IDEAS YET.</p>') + '</div>')
    return "".join(blocks)


def header_state() -> Dict[str, Any]:
    a = S.agent
    hist = a.history if a else []
    last = next((r for r in reversed(hist) if r.insight), None)
    err = S.error or (a.last_error if a else "") or (S.engine.last_error if S.engine else "")
    if a and a.budget.spent >= a.budget.limit:
        err = f"BUDGET OF ${a.budget.limit:.0f} REACHED"
    if err:
        status = "ERROR · " + err
    elif S.recording:
        status = "LISTENING · " + SOURCES[S.source]
    elif S.feed_task:
        status = "STREAMING TEXT"
    else:
        status = "READY · " + SOURCES[S.source]
    level = 0
    if S.engine is not None:
        level = min(100, int(S.engine.level * 2000))
    return {
        "status": status, "err": bool(err), "live": S.recording or bool(S.feed_task),
        "recording": S.recording, "source": S.source, "lang": S.lang, "speaker": S.speaker,
        "streaming": bool(S.feed_task),
        "model": S.provider.label if S.provider else "—",
        "latency": f"LATENCY {last.latency_s:.1f} S" if last else "LATENCY —",
        "queue": f"QUEUE {a.inbox.qsize() if a else 0}",
        "count": f"UTTERANCES {len(S.segs)}",
        "spend": f"SPEND ${a.budget.spent:.3f} / ${BUDGET_USD:.0f}" if a else "SPEND —",
        "level": level,
    }


def export_markdown() -> str:
    a = S.agent
    res = results_by_uid()
    lines = ["# 0xmoira session report", "", f"Model: {S.provider.label if S.provider else '-'}", ""]
    syn = a.synthesis if a else None
    if syn:
        lines += [f"## {syn.headline}", "", syn.situation, "", f"**Next move:** {syn.recommendation}", "",
                  f"**Dynamics:** {syn.dynamics}", "", f"**Undercurrent:** {syn.undercurrent}", "", f"**Risks:** {syn.risks}", ""]
        if syn.tells:
            lines += ["### Tells", ""] + [f"- **{t.who}:** \"{t.quote}\" — {t.pattern}" for t in syn.tells] + [""]
        lines += ["### Speakers", ""]
        for p in syn.speakers:
            lines += [f"#### {p.label} · {p.role}", ""] + [f"- **{k}:** {v}" for k, v in (
                ("Evidence", p.evidence), ("State", p.state), ("Motives", p.motives), ("Defences", p.defenses),
                ("Relation", p.relation), ("Subtext", p.subtext), ("Bottom line", p.bottom_line), ("Lever", p.lever),
                ("Credibility", p.credibility)) if v] + [""]
    lines += ["## Transcript", ""]
    for s in S.segs:
        m, sec = divmod(int(s.t), 60)
        lines.append(f"**[{m:02d}:{sec:02d}] {s.speaker}:** {s.text}")
        r = res.get(s.uid)
        if r and r.insight:
            ins = r.insight
            lines.append(f"> tension {ins.cognitive_tension_score}/10 · {ins.mood} · {ins.evidence}  ")
            lines.append(f"> next move ({ins.intervention_type}): {ins.tactical_action}")
            for f in ins.flags:
                lines.append(f"> flag {f.category}/{f.severity}: \"{f.phrase}\" — {f.reason}")
        lines.append("")
    if a:
        lines.append(f"_Spend: ${a.budget.spent:.4f}_")
    return "\n".join(lines)


async def run_feed(text: str, wpm: int) -> None:
    pattern = re.compile(r"^\s*([^:\n]{1,32}):\s+(.+)$")
    try:
        for raw in text.splitlines():
            if not raw.strip():
                continue
            m = pattern.match(raw)
            speaker, line = (m.group(1).strip(), m.group(2)) if m else (S.speaker, raw)
            await asyncio.sleep(max(0.4, len(line.split()) / max(30, wpm) * 60))
            ingest(speaker, line)
    finally:
        S.feed_task = None


# =========================================================================== #
# Main page
# =========================================================================== #
def header_html() -> str:
    src = "".join(f'<option value="{k}">{e(v)}</option>' for k, v in SOURCES.items())
    langs = "".join(f'<option value="{k}">{k.upper()}</option>' for k in LANGS)
    chips = "".join(f'<span class="spk spk-{i}" data-spk="{c}" title="Current speaker {c}" onclick="oxm.cmd(\'speaker\',\'{c}\')">{c}</span>'
                    for i, c in enumerate("ABCD"))
    return f"""
<header class="bar">
  <h1>0X<br>MOIRA</h1>
  <div class="status"><span class="dot" id="oxm-dot"></span><span id="oxm-status">CONNECTING…</span></div>
  <div class="controls">
    <label>SOURCE <select id="oxm-source" onchange="oxm.cmd('source', this.value)">{src}</select></label>
    <label>LANGUAGE <select id="oxm-lang" onchange="oxm.cmd('lang', this.value)">{langs}</select></label>
    <span class="chips" id="oxm-chips" title="Who is speaking now (browser mic has no speaker separation)">{chips}</span>
    <button class="btn btn-rec" id="oxm-rec" onclick="oxm.cmd('rec')">● REC</button>
    <button class="btn" onclick="oxm.cmd('export')">EXPORT</button>
    <button class="btn" onclick="oxm.cmd('clear')">CLEAR</button>
    <button class="btn" title="Invert colours" onclick="oxm.invert()">◐</button>
    <button class="btn" onclick="oxm.cmd('signout')">SIGN OUT</button>
  </div>
</header>"""


FEED_HTML = f"""
<div class="feed hidden" id="oxm-feed">
  <div class="panel-head"><span>TEXT INPUT</span>
    <span><button class="btn btn-small" onclick='oxm.sample({json.dumps(SAMPLE)})'>LOAD SAMPLE</button></span></div>
  <textarea id="oxm-feed-text" rows="5" placeholder="Paste a transcript. One line per turn, optionally 'Name: text'. It is streamed line by line into the analysis."></textarea>
  <div class="feed-controls">
    <label>WORDS/MIN <input type="range" id="oxm-wpm" min="60" max="400" step="10" value="160"
      oninput="document.getElementById('oxm-wpm-val').textContent=this.value"> <span id="oxm-wpm-val">160</span></label>
    <button class="btn" id="oxm-stream" onclick="oxm.stream()">▶ STREAM</button>
  </div>
</div>"""

FOOTER_HTML = """
<footer class="bar foot">
  <span id="f-model">—</span><span id="f-latency">LATENCY —</span><span id="f-queue">QUEUE 0</span>
  <span id="f-count">UTTERANCES 0</span><span id="f-spend">SPEND —</span>
  <span class="level" title="Server-mic input level"><i id="f-level"></i></span>
</footer>"""


@ui.page("/")
def index() -> None:
    if S.provider is None or S.agent is None:
        ui.navigate.to("/login")
        return
    head()
    ui.page_title("0xmoira")
    client_id = ui.context.client.id

    with ui.element("div").classes("oxm"):
        ui.html(header_html(), sanitize=False)
        with ui.element("main").classes("grid"):
            with ui.element("section").classes("panel panel-transcript"):
                t_head = ui.html("", sanitize=False)
                transcript = ui.html("", sanitize=False).classes("transcript")
                ui.html('<p class="interim" id="oxm-interim"></p>', sanitize=False)
                ui.html(FEED_HTML, sanitize=False)
            with ui.element("section").classes("panel panel-overall"):
                overall = ui.html("", sanitize=False).style("display:flex;flex-direction:column;min-height:0;flex:1 1 auto")
            with ui.element("aside").classes("panel panel-side"):
                side = ui.html("", sanitize=False)
        ui.html(FOOTER_HTML, sanitize=False)

    last: Dict[str, Any] = {"t": None, "o": None, "s": None, "h": None}

    def tick() -> None:
        a = S.agent
        if a is None:
            return
        n_hist = len(a.history)
        kt = (S.version, n_hist)
        if kt != last["t"]:
            transcript.content = render_transcript()
            t_head.content = (f'<div class="panel-head"><span>TRANSCRIPT</span>'
                              f'<span class="muted">{len(S.segs)} SEGMENTS</span></div>')
            last["t"] = kt
        ko = (a.synth_version, a.synth_running, a.synth_error, S.version)
        if ko != last["o"]:
            overall.content = render_overall()
            last["o"] = ko
        ks = (n_hist, S.version)
        if ks != last["s"]:
            side.content = render_side()
            last["s"] = ks
        hs = header_state()
        if hs != last["h"]:
            ui.run_javascript(f"oxm.apply({json.dumps(hs)})")
            last["h"] = hs

    async def on_cmd(ev: Any) -> None:
        args = event_args(ev)
        cmd, value = args.get("cmd"), args.get("value")
        try:
            if cmd == "rec":
                if S.recording:
                    await stop_recording()
                elif S.source == "browser":
                    S.recording, S.rec_client, S.error = True, client_id, ""
                    ui.run_javascript(f"oxm.speech.start({json.dumps(S.lang)})")
                elif S.source in ("gemini", "endpoint"):
                    S.error = ""
                    await start_server_mic()
                    S.recording = True
            elif cmd == "source" and value in SOURCES:
                if S.recording:
                    await stop_recording()
                S.source = value
            elif cmd == "lang" and value in LANGS:
                S.lang = value
                if S.recording and S.source == "browser":
                    ui.run_javascript(f"oxm.speech.stop(); oxm.speech.start({json.dumps(value)})")
            elif cmd == "speaker" and isinstance(value, str):
                S.speaker = value
            elif cmd == "analyse":
                if S.agent and not S.agent.request_synthesis():
                    ui.notify("Nothing to analyse yet, or an analysis is already running.")
            elif cmd == "export":
                ui.download.content(export_markdown(), f"oxmoira-{time.strftime('%Y%m%d-%H%M')}.md", "text/markdown")
            elif cmd == "clear":
                if S.recording:
                    await stop_recording()
                await new_session()
            elif cmd == "signout":
                await sign_out()
                ui.navigate.to("/login")
                return
            elif cmd == "stream":
                if S.feed_task:
                    await stop_feed()
                else:
                    text = (value or {}).get("text", "")
                    if not text.strip():
                        ui.notify("Paste a transcript or load the sample first.")
                    else:
                        S.feed_task = asyncio.create_task(run_feed(text, int((value or {}).get("wpm", 160))))
        except Exception as exc:
            log.exception("command %s failed", cmd)
            S.error = f"{type(exc).__name__}: {exc}"
            S.recording = False
        tick()

    def on_final(ev: Any) -> None:
        args = event_args(ev)
        conf = args.get("confidence") or 0
        ingest(S.speaker, str(args.get("text", "")), int(conf * 100) if conf else None)

    def on_speech_error(ev: Any) -> None:
        err = event_args(ev).get("error", "unknown")
        if err == "not-supported":
            S.error = "This browser has no speech recognition. Use Chrome, Edge or Safari, or another source."
        elif err in ("not-allowed", "service-not-allowed"):
            S.error = "Microphone or speech recognition was blocked in the browser."
        else:
            S.error = f"Speech recognition: {err}"
        if err in ("not-supported", "not-allowed", "service-not-allowed"):
            S.recording = False

    ui.on("oxm_cmd", on_cmd)
    ui.on("oxm_final", on_final)
    ui.on("oxm_speech_error", on_speech_error)
    ui.timer(0.25, tick)
    ui.timer(0.6, lambda: ui.run_javascript("oxm.init()"), once=True)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="0xmoira", host="127.0.0.1", port=int(os.getenv("OXMOIRA_PORT", "8080")),
           reload=False, show=os.getenv("OXMOIRA_NO_BROWSER") != "1", favicon="◐")
