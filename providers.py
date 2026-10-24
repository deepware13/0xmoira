"""LLM providers (Anthropic, Gemini), sign-in verification, credential storage and Gemini transcription.

Both providers expose the same call: `json_call(system, schema, user_msg, max_tokens, effort)` returning
(parsed dict or None, cost in USD, error text). The schema is a strict JSON Schema (every object has
`additionalProperties: false` and lists all keys as required; no numeric/length constraints), which both
Anthropic structured outputs and Gemini `responseJsonSchema` accept. Range checks happen in agent_core.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import httpx

log = logging.getLogger("oxmoira.providers")

CRED_DIR = Path(os.getenv("OXMOIRA_HOME", str(Path.home() / ".oxmoira")))
CRED_FILE = CRED_DIR / "credentials.json"

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"

# USD per million tokens: (input, output, cache read). Cache writes are billed at 1.25x input.
ANTHROPIC_PRICES: Dict[str, Tuple[float, float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-fable-5": (10.0, 50.0, 1.0),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-opus-4-7": (5.0, 25.0, 0.50),
    "claude-opus-4-6": (5.0, 25.0, 0.50),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}
UNKNOWN_PRICE = (5.0, 25.0, 0.50)  # conservative for models not in the table
# Gemini prices change often and differ per model; these are deliberately conservative estimates.
GEMINI_PRICE = (float(os.getenv("OXMOIRA_GEMINI_PRICE_IN", "0.50")), float(os.getenv("OXMOIRA_GEMINI_PRICE_OUT", "3.00")))

# Models that accept server-side refusal fallbacks in the `"default"` form on the Claude API.
FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LoginError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def load_credentials() -> Dict[str, Any]:
    try:
        data = json.loads(CRED_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_credentials(data: Dict[str, Any]) -> None:
    CRED_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = CRED_FILE.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, CRED_FILE)
    os.chmod(CRED_FILE, 0o600)


def clear_credentials() -> None:
    try:
        CRED_FILE.unlink()
    except FileNotFoundError:
        pass


def mask_key(key: str) -> str:
    return f"{key[:7]}…{key[-4:]}" if len(key) > 14 else "•" * len(key)


# --------------------------------------------------------------------------- #
# Schema helper
# --------------------------------------------------------------------------- #
_UNSUPPORTED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                "minLength", "maxLength", "minItems", "maxItems", "pattern"}


def strict_schema(schema: Any) -> Any:
    """Strip constraints structured outputs do not support; close every object and require all its keys."""
    if isinstance(schema, list):
        return [strict_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: strict_schema(v) for k, v in schema.items() if k not in _UNSUPPORTED}
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"].keys())
    return out


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    return json.loads(text)


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
def anthropic_price(model: str) -> Tuple[float, float, float]:
    return ANTHROPIC_PRICES.get(model, UNKNOWN_PRICE)


async def verify_anthropic(api_key: str) -> List[str]:
    client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=1, timeout=20.0)
    try:
        ids = [m.id async for m in client.models.list(limit=100)]
    except anthropic.AuthenticationError as exc:
        raise LoginError("Anthropic rejected this API key.") from exc
    except anthropic.PermissionDeniedError as exc:
        raise LoginError("This Anthropic key lacks permission to list models.") from exc
    except anthropic.APIConnectionError as exc:
        raise LoginError("Cannot reach api.anthropic.com. Check the network.") from exc
    except anthropic.APIStatusError as exc:
        raise LoginError(f"Anthropic error {exc.status_code}: {exc.message}") from exc
    finally:
        await client.close()
    if not ids:
        raise LoginError("Key accepted, but no models are available to it.")
    return ids


def default_anthropic_model(ids: List[str]) -> str:
    if DEFAULT_ANTHROPIC_MODEL in ids:
        return DEFAULT_ANTHROPIC_MODEL
    sonnets = [i for i in ids if "sonnet" in i]
    return (sonnets or ids)[0]


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=2, timeout=60.0)
        self._fallbacks = model in FALLBACK_MODELS
        self._effort = True

    @property
    def label(self) -> str:
        return f"ANTHROPIC · {self.model}"

    def price_in_out(self) -> Tuple[float, float]:
        p = anthropic_price(self.model)
        return p[0], p[1]

    def _cost(self, model: str, usage: Any) -> float:
        p_in, p_out, p_read = anthropic_price(model)
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        return (inp * p_in + out * p_out + cw * p_in * 1.25 + cr * p_read) / 1_000_000

    async def json_call(self, system: str, schema: Dict[str, Any], user_msg: str, max_tokens: int,
                        effort: str) -> Tuple[Optional[Dict[str, Any]], float, str]:
        output_config: Dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self._effort:
            output_config["effort"] = effort
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_msg}],
            output_config=output_config,
        )
        try:
            if self._fallbacks:
                msg = await self.client.beta.messages.create(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                msg = await self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as exc:
            text = str(exc).lower()
            if self._fallbacks and "fallback" in text:
                self._fallbacks = False
                return await self.json_call(system, schema, user_msg, max_tokens, effort)
            if self._effort and "effort" in text:
                self._effort = False
                return await self.json_call(system, schema, user_msg, max_tokens, effort)
            raise
        cost = self._cost(getattr(msg, "model", self.model) or self.model, msg.usage)
        if msg.stop_reason == "refusal":
            return None, cost, "model declined (refusal)"
        if msg.stop_reason == "max_tokens":
            return None, cost, "output truncated (max_tokens)"
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        try:
            data = _extract_json(text)
        except ValueError as exc:
            return None, cost, f"invalid JSON: {exc}"
        return (data if isinstance(data, dict) else None), cost, "" if isinstance(data, dict) else "JSON is not an object"

    async def close(self) -> None:
        await self.client.close()


# --------------------------------------------------------------------------- #
# Gemini (REST)
# --------------------------------------------------------------------------- #
def _gemini_version_key(name: str) -> Tuple[float, int]:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    ver = float(m.group(1)) if m else 0.0
    return ver, -len(name)


async def verify_gemini(api_key: str) -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{GEMINI_BASE}/models", params={"pageSize": 1000},
                                    headers={"x-goog-api-key": api_key})
    except httpx.HTTPError as exc:
        raise LoginError("Cannot reach generativelanguage.googleapis.com. Check the network.") from exc
    if resp.status_code in (400, 401, 403):
        raise LoginError("Google rejected this Gemini API key.")
    if resp.status_code != 200:
        raise LoginError(f"Gemini error {resp.status_code}: {resp.text[:200]}")
    ids = []
    for m in resp.json().get("models", []):
        name = str(m.get("name", "")).removeprefix("models/")
        if "generateContent" in m.get("supportedGenerationMethods", []) and name.startswith("gemini"):
            if not any(x in name for x in ("embedding", "tts", "image", "live", "audio")):
                ids.append(name)
    if not ids:
        raise LoginError("Key accepted, but no Gemini text models are available to it.")
    return sorted(ids, key=_gemini_version_key, reverse=True)


def default_gemini_model(ids: List[str]) -> str:
    flash = [i for i in ids if "flash" in i and "lite" not in i and "preview" not in i and "exp" not in i]
    return (flash or [i for i in ids if "flash" in i] or ids)[0]


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._key = api_key
        self.client = httpx.AsyncClient(timeout=60.0)
        self._json_schema_ok = True

    @property
    def label(self) -> str:
        return f"GEMINI · {self.model}"

    def price_in_out(self) -> Tuple[float, float]:
        return GEMINI_PRICE

    def _cost(self, usage: Dict[str, Any]) -> float:
        inp = usage.get("promptTokenCount", 0) or 0
        out = (usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0)
        return (inp * GEMINI_PRICE[0] + out * GEMINI_PRICE[1]) / 1_000_000

    async def generate(self, system: str, parts: List[Dict[str, Any]], schema: Dict[str, Any],
                       max_tokens: int, model: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], float, str]:
        gen: Dict[str, Any] = {"responseMimeType": "application/json", "maxOutputTokens": max_tokens}
        parts = list(parts)
        if self._json_schema_ok:
            gen["responseJsonSchema"] = schema
        else:
            parts.append({"text": "Return JSON matching this JSON Schema exactly:\n" + json.dumps(schema)})
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": gen,
        }
        url = f"{GEMINI_BASE}/models/{model or self.model}:generateContent"
        resp = await self.client.post(url, json=body, headers={"x-goog-api-key": self._key})
        if resp.status_code == 400 and self._json_schema_ok and "responseJsonSchema" in resp.text:
            self._json_schema_ok = False
            return await self.generate(system, parts, schema, max_tokens, model)
        if resp.status_code != 200:
            return None, 0.0, f"Gemini HTTP {resp.status_code}: {resp.text[:300]}"
        payload = resp.json()
        cost = self._cost(payload.get("usageMetadata", {}))
        cands = payload.get("candidates") or []
        if not cands:
            block = payload.get("promptFeedback", {}).get("blockReason", "no candidates")
            return None, cost, f"Gemini returned nothing ({block})"
        cand = cands[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        if cand.get("finishReason") == "MAX_TOKENS":
            return None, cost, "output truncated (MAX_TOKENS)"
        try:
            data = _extract_json(text)
        except ValueError as exc:
            return None, cost, f"invalid JSON ({cand.get('finishReason')}): {exc}"
        return (data if isinstance(data, dict) else None), cost, "" if isinstance(data, dict) else "JSON is not an object"

    async def json_call(self, system: str, schema: Dict[str, Any], user_msg: str, max_tokens: int,
                        effort: str) -> Tuple[Optional[Dict[str, Any]], float, str]:
        return await self.generate(system, [{"text": user_msg}], schema, max(max_tokens, 8192))

    async def close(self) -> None:
        await self.client.aclose()


# --------------------------------------------------------------------------- #
# Gemini transcription (audio chunk -> diarized segments)
# --------------------------------------------------------------------------- #
TRANSCRIBE_SYSTEM = (
    "You are a meeting transcription engine. Transcribe the audio verbatim in the language actually spoken "
    "(English, German or mixed); never translate, never summarise, never add words that were not said. "
    "Split it into segments at every change of speaker. Label speakers with single capital letters. "
    "You are given the labels used so far with a recent line from each: reuse a label only when the voice and "
    "the conversational context clearly match that speaker, otherwise use the next unused letter. "
    "If the audio holds no intelligible speech (silence, noise, music, a cough), return an empty segments list. "
    "Respond only with JSON matching the schema."
)
TRANSCRIBE_SCHEMA = strict_schema({
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {"type": "object", "properties": {"speaker": {"type": "string"}, "text": {"type": "string"}}},
        }
    },
})


class GeminiTranscriber:
    """Callable for AudioEngine: (seq, wav bytes) -> {"segments": [...]}; reports cost through `on_cost`."""

    def __init__(self, provider: GeminiProvider, model: Optional[str] = None, on_cost=None) -> None:
        self.provider = provider
        self.model = model
        self.on_cost = on_cost
        self.recent: Dict[str, str] = {}

    async def __call__(self, seq: int, wav: bytes) -> Dict[str, Any]:
        known = "\n".join(f"{k}: \"{v[-80:]}\"" for k, v in self.recent.items()) or "(none yet)"
        parts = [
            {"text": f"Speaker labels used so far:\n{known}\n\nTranscribe this audio chunk."},
            {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav).decode("ascii")}},
        ]
        data, cost, err = await self.provider.generate(TRANSCRIBE_SYSTEM, parts, TRANSCRIBE_SCHEMA, 2048, self.model)
        if self.on_cost:
            self.on_cost(cost)
        if data is None:
            raise RuntimeError(err)
        for seg in data.get("segments", []) or []:
            spk, text = str(seg.get("speaker", "")).strip(), str(seg.get("text", "")).strip()
            if spk and text:
                self.recent[spk] = text
        return data
