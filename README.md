# 0xmoira: real-time psychological meeting copilot

Built at the AI Hackathon Berlin (Jamie).

0xmoira listens to a meeting, transcribes it, and runs an analysis loop over a rolling window of the conversation. For every utterance it shows a cue-based read of the speaker (tension, mood, dominance, defensiveness, flagged phrases, ideas) and **your next move**. Every few utterances it writes an overall analysis: speaker profiles, dynamics, undercurrent, risks and the single most useful next step.

The interface ports the strict black-and-white design of the 0xmoira prototype to a Python/NiceGUI app.

## Quick start

```bash
git clone https://github.com/deepware13/0xmoira.git
cd 0xmoira
python3.12 -m venv .venv          # Python 3.11+
.venv/bin/pip install -r requirements.txt
.venv/bin/python main_ui.py       # opens http://127.0.0.1:8080
```

Sign in with an **Anthropic** or **Google Gemini** API key. The key is verified against the provider (model list), you pick the analysis model, and optionally save it to `~/.oxmoira/credentials.json` (owner-only permissions). Keys are never sent back to the browser. `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` in the environment can be used with one click.

## Input sources

| Source | Needs | Speaker separation |
|---|---|---|
| **Browser mic** (default) | Chrome, Edge or Safari (Web Speech API; the browser vendor does the speech recognition) | Click the A–D badges in the header when the speaker changes |
| **Server mic · Gemini ASR** | A Gemini key (as main provider, or the optional second key on the sign-in screen) | Gemini labels speakers per chunk |
| **Server mic · ASR endpoint** | `OXMOIRA_ASR_URL` (+ `OXMOIRA_ASR_KEY`) returning `{utterance_id, speaker, text, duration_ms}` | From the endpoint |
| **Text** | Nothing | `Name: text` per line; **LOAD SAMPLE** gives a demo dialogue |

Server-mic sources capture `OXMOIRA_AUDIO_DEVICE` (default `BlackHole 2ch`, falls back to the default input) and cut utterances at 450 ms pauses (8 s cap).

## Architecture

* `providers.py`: Anthropic (official SDK, structured outputs via `output_config.format`, server-side refusal fallback on supported models) and Gemini (REST, `responseJsonSchema`); sign-in verification; credential store; Gemini audio transcription.
* `agent_core.py`: per-utterance analysis over a rolling window of 10 utterances plus each speaker's smoothed baseline; periodic overall synthesis as a separate background task. Quotes, flags and ideas the model returns are checked against the real transcript and dropped if not found. A persisted spend guard stops new calls at `OXMOIRA_BUDGET_USD` (default 75).
* `audio_engine.py`: PortAudio capture, energy VAD endpointing, ordered concurrent transcription.
* `main_ui.py`: NiceGUI app (sign-in, three-panel dashboard, export to Markdown). Everything runs on one asyncio loop; the UI polls append-only state every 250 ms and never blocks on the LLM.

## Configuration

| Variable | Default | |
|---|---|---|
| `OXMOIRA_BUDGET_USD` | `75` | Hard spend cap (estimated from token usage) |
| `OXMOIRA_SYNTH_EVERY` | `6` | Analysed utterances between overall analyses (`0` = manual) |
| `OXMOIRA_GEMINI_PRICE_IN` / `_OUT` | `0.50` / `3.00` | Gemini $/M tokens used for the estimate |
| `OXMOIRA_PORT` | `8080` | |
| `OXMOIRA_AUDIO_DEVICE` | `BlackHole 2ch` | Server-mic input |
| `OXMOIRA_ASR_URL`, `OXMOIRA_ASR_KEY`, `OXMOIRA_ASR_MODEL` | | Custom ASR endpoint |

## Limitations

* Readings are model inferences from words only, not facts about a person; treat them as prompts for attention. No diagnoses are produced.
* Inferring emotions of people at work is prohibited in the EU (AI Act Art. 5(1)(f)); do not use this in that context.
* Browser speech recognition has no speaker separation; Gemini speaker labels may not stay consistent across chunks.
* Spend is an estimate from token usage; check your provider's billing page.
