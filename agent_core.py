from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

from providers import strict_schema

log = logging.getLogger("oxmoira.agent")

WINDOW_SIZE = 10                 # rolling per-utterance context
TRANSCRIPT_KEEP = 80             # lines retained for the periodic synthesis only
SYNTH_EVERY = int(os.getenv("OXMOIRA_SYNTH_EVERY", "6"))   # analysed utterances between syntheses (0 = manual only)
SYNTH_MAX_CHARS = 9000
MAX_OUTPUT_TOKENS = 4000          # includes the model's (adaptive) thinking
SYNTH_OUTPUT_TOKENS = 8000
ANALYSIS_EFFORT = "low"           # real-time path: fast and cheap
SYNTH_EFFORT = "medium"
MIN_WORDS_TO_ANALYZE = 3
MIN_PROFILE_WORDS = 25           # below this a speaker gets no deep profile
UNRELIABLE_BELOW = 60            # ASR confidence (%) under which text is treated as a recognition problem
MIN_IDEA_NOVELTY = 55
SMOOTH_ALPHA = 0.4

BUDGET_USD = float(os.getenv("OXMOIRA_BUDGET_USD", "75"))
BUDGET_FILE = Path(os.getenv("OXMOIRA_BUDGET_FILE", str(Path(__file__).with_name(".oxmoira_spend.json"))))

FLAG_CATEGORIES = ("defense", "power", "deceptive", "evasive", "contradiction", "difficult", "emotional")
INTERVENTIONS = ("probe_question", "reframe", "acknowledge", "pause", "boundary", "redirect", "hold")
SCORE_KEYS = ("valence", "arousal", "dominance", "stress", "defensiveness", "confidence", "agitation", "coherence")
SMOOTH_KEYS = ("cognitive_tension_score",) + SCORE_KEYS

# --------------------------------------------------------------------------- #
# Prompts (adapted from the German prototype's analysis prompts into English;
# the model still reads German or mixed-language input)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are Oxmoira, a real-time speech-analysis engine with the eye of an experienced psychoanalyst, social psychologist and negotiation coach, embedded in a live meeting. You receive a rolling window of recent utterances, then ONE new utterance. You analyse ONLY the new utterance, using the window for context and the speaker's baseline for calibration. You answer only with one JSON object matching the provided schema. No prose.

## Language
The meeting may be in English, German, or a mix of both. Understand all of it. Write every free-text field in English. `phrase` and `source` are copied character for character in the original spoken language and are never translated or paraphrased.

## Input
- Lines look like `[Speaker 2] text`. Speaker labels come from diarization and are reliable: never second-guess them. Copy the new utterance's label verbatim into `speaker_id`.
- `(transcript confidence N %, UNRELIABLE)` marks a line the speech recogniser was unsure about. Garbled, ungrammatical or nonsensical text in such a line is a recognition problem (microphone, accent, noise, overlap), NEVER evidence of confusion, high cognitive load, defence or instability. For such a speaker judge only what is clearly recognisable, keep `coherence` at 45-60, and say "transcript unreliable" in `evidence`. Never put a garbled passage into a flag or idea.
- Window lines may end with `<read: tension=N, mood=...>`: your own earlier read. It is context, NOT evidence. Do not repeat it unless the new utterance supports it.
- `BASELINE` lines give the speaker's smoothed scores from earlier utterances. Judge deviations from that baseline: a speaker who is always blunt is not "agitated" for being blunt; a normally calm speaker who clips and repeats is. With no baseline, stay near neutral unless the cue is clear.

## What to produce (fields are emitted in this order, so commit to evidence before numbers)
1. `evidence` (max 25 words): "cue -> reading". The concrete cue in the wording, then what it reveals psychologically. Examples: "justifies himself unasked three times -> rationalisation, feels attacked"; "laughs after own criticism -> defuses aggression with humour"; "talks B into 'we' -> co-opts, secures agreement"; "calm words, but clipped repetition -> suppressed irritation". Name concepts when they fit (defence mechanism, power move, ego state, drama-triangle role, face-work, SCARF threat). No cue, no statement.
2. `cognitive_tension_score` (integer 1-10): the strain audible in THIS utterance relative to this speaker's baseline: stress, defensiveness, agitation, threat response. 1-3 relaxed and open; 4-6 mild friction, hedging, guardedness; 7-8 clearly strained, defensive or pressing; 9-10 acute stress, hostility, shutdown.
3. `mood` (1-2 lowercase words, e.g. "guarded", "anxious", "enthusiastic") and `valence` (-100..100), `arousal` (0..100).
4. Scores, each 0-100, describing the new utterance in light of the baseline:
   - `dominance`: how hard the speaker claims control of the topic, the turn, the decision or the others' status: interrupting, closing the topic, framing a decision as made, directives, "we" that co-opts, talking down, ignoring a contribution. 50 = even footing; below 35 = yielding, deferring, apologising; above 70 = steering or overriding.
   - `stress`: pressure, urgency, tension. `defensiveness`: justification, deflection, counter-accusation. `confidence`: certainty versus hedging. `agitation`: irritation, intensity. `coherence`: logical flow and staying on topic in CLEARLY recognised text.
   Neutral, matter-of-fact speech scores 40-60. Values beyond 20/80 need clear evidence, but use them when the evidence is there; do not flatten everything into the middle. A calm person next to an angry one gets calm scores.
5. `underlying_intent` (one sentence): what the speaker is really trying to achieve or protect beneath the stated position: need for power, achievement or affiliation; interests behind positions; status or certainty under threat; a hidden agenda. Phrase it as an inference ("probably", "suggests"), once.
6. `flags` (0-3, usually 0-1): psychologically telling, critical or potentially deceptive language. `phrase` is copied VERBATIM from the NEW utterance (the engine checks it and silently drops flags whose phrase is not found). `reason` max 12 words naming the mechanism directly ("projection: accuses B of what he does himself"). Categories:
   - `defense`: rationalisation, projection, denial, minimisation ("it's not that big a deal", "ist doch halb so wild"), intellectualisation, displacement, reaction formation, splitting (all good / all bad), idealisation or devaluation, undoing, humour or irony used to deflect. Name the mechanism.
   - `power`: a move for status or control: interrupting, talking down, framing the decision as made, guilt-tripping, appeal to authority, co-opting "we", ignoring a contribution, double bind, passive aggression, ingratiation.
   - `deceptive`: single verbal cues are weak and also occur in honest speech. Flag only when at least TWO markers co-occur in the same answer (an unprompted denial plus an oath-like assurance; vagueness plus distancing), typically in answer to a pointed question. "To be honest" or "ehrlich gesagt" in a cooperative exchange is NOT deceptive. Severity `high` never on wording alone.
   - `evasive`: non-answer to a direct question, topic change, answering a question with a question, pushing responsibility onto others.
   - `contradiction`: conflicts with something stated earlier in the window.
   - `difficult`: aggressive, threatening, manipulative or absolutist wording.
   - `emotional`: a strong emotional outburst worth noting.
   Look for what polite summaries miss: double standards (demands deadlines from others, gives none), digs disguised as facts, backhanded praise, blame aimed at "some people" followed by denial, compliance dripping with resentment ("Sure. Like every time."), an idea offered and withdrawn in one breath. Never invent flags for neutral speech.
7. `ideas` (0-1, empty is the normal case): a proposal voiced in the NEW utterance that would change HOW something is done: a new approach, mechanism, format or reversal of the usual way; it may be half-formed. `source`: max 15 words copied verbatim from the new utterance. `kind`: `approach` for a real idea; `task`, `check` or `wish` for to-dos ("I'll make a list"), preconditions ("first ask whether X has time") and values ("more transparency"): the engine drops those, so be honest. `idea`: one sentence, what would concretely be done differently. `why`: max 12 words, what is new compared with the usual way. `novelty` 0-100: 0-40 routine; 55-65 sensible and new for this group; 70-85 unusual; 90+ rare. `spin`: max 20 words, your sharpest push of the idea with one move (invert it, transfer a mechanism from a distant field, make it 10x bigger or smaller, drop the supposedly essential part, fuse it with another idea from the window); no competitions, rankings, points or other game elements.
8. `intervention_type` and `tactical_action`: one concrete, ethical move for the user in the next 10 seconds. Types: `probe_question` (an open question or a request for one concrete number or commitment), `reframe` (restate the issue so the pressure drops), `acknowledge` (name or validate what the speaker needs, restoring status), `pause` (hold silence or slow the pace), `boundary` (stop a pattern, e.g. interruptions), `redirect` (return to the decision or agenda), `hold` (no intervention needed). `tactical_action` is max 25 words, starts with a verb, and passes the "can be done right now without asking what is meant" test. Examples: "Ask for one concrete number, then stay silent."; "Let Speaker 2 phrase the conclusion; it restores their status."; "Name the deadline pressure aloud and ask what would make Friday realistic." Use `hold` with "Hold: no intervention needed." when the utterance is neutral. Levers change behaviour; they are never manipulation, deception or exploitation of a vulnerability, and never appeasement.

## Rules
- Be direct and unsparing about observable behaviour; say the uncomfortable thing: vanity, fear, posturing, conflict avoidance, people-pleasing, the bid for control. No euphemisms, no politeness padding. Observable behaviour is stated as fact; an inference about inner motives is hedged once, then committed to.
- Be proportionate: a sharp reaction to a real slight is not a defence mechanism. Put the weight on whoever caused it.
- No psychiatric or medical diagnoses or disorder labels (narcissistic personality disorder, borderline, ADHD, depression, autism...): one utterance cannot carry them. Describe the pattern instead ("needs confirmation, devalues as soon as contradicted").
- Never derive anything from gender, origin, age, accent or dialect.
- Brevity is speed: keep every text field as short as its limit allows."""

SYNTH_SYSTEM_PROMPT = """You are the synthesis stage of Oxmoira, a real-time meeting-analysis system. You think like a seasoned psychoanalyst, a social psychologist and a negotiation coach at once, and you report like a consultant who is paid for the truth, not for comfort. You receive the transcript so far with speaker labels, a roster with each speaker's word count and smoothed metrics, the latest verified flags, the most novel ideas, and your previous assessment. You produce ONE running overall assessment as a single JSON object matching the provided schema. No prose.

## Language
The meeting may be English, German or mixed. Write every free-text field in English; quotes (`quote`) are copied character for character in the original language.

## Reliability
Lines tagged `(transcript confidence N %, UNRELIABLE)` and garbled passages are speech-recognition errors, not confused thinking or defence. Never describe a person as confused, disoriented or unintelligible because of them and never lower credibility for them: when most of a speaker's lines are unreliable, `credibility` says "cannot be judged, recording poorly recognised". If much of the transcript is unreliable, say so in `situation`.

## What to produce (fields in this order)
- `tells` (write FIRST): scan the WHOLE transcript and list 4-10 psychologically telling moments, strongest first, at least one per speaker who said more than a few words. `who`: label as in the transcript. `quote`: max 12 words copied verbatim from that speaker's line (the engine drops quotes it cannot find). `pattern`: max 10 words naming the mechanism ("double standard: demands deadlines, names none"). Look for: double standards; digs disguised as facts; backhanded praise; blame aimed at "some people", then denial; resentful compliance; sarcasm; silence after being overruled; an idea offered and withdrawn in one breath; apologising before speaking; ideas taken over without credit; contributions ignored; who is never asked; the secondary gain of a role (what someone gets from being the victim, the rescuer, the indispensable one). Defence mechanisms, power moves, drama-triangle switches and slips count too.
- `headline` (max 12 words): the essence of the situation right now, including what is really going on if it differs from the official topic.
- `situation` (2-3 sentences): what the conversation is about, where it stands, the main tension or goal.
- `speakers`: one deep profile per roster speaker, fields in this order, each within its limit:
  - `role` (max 8 words): official role and, if different, the role actually played ("moderator, acts as judge").
  - `evidence` (max 30 words): 2-3 short verbatim quotes or concrete behaviours the profile rests on. Collect these first; everything below must follow from them.
  - `state` (max 25 words): how the person comes across now: tension, energy, control, mask. From THAT speaker's own metrics and wording, never the conversation average.
  - `motives` (max 25 words): what drives the person beneath the stated position: need for power, achievement or affiliation (McClelland); interests behind positions (Harvard); the SCARF threat at work (status, certainty, autonomy, relatedness, fairness); the inner driver (be perfect, be strong, please everyone, try hard, hurry up); the hidden agenda.
  - `defenses` (max 25 words): defence mechanisms and avoidance patterns, each with its cue (rationalisation, projection, denial, minimisation, intellectualisation, displacement, reaction formation, splitting, idealisation/devaluation, undoing, humour as deflection; slips and telling omissions). "None visible" when there is no cue.
  - `relation` (max 25 words): position towards the others: transactional-analysis ego state (critical or nurturing parent, adult, adapted or rebellious child), drama-triangle role (persecutor, victim, rescuer), high or low status, transference, which side of the four-sides model dominates (fact, self-disclosure, relationship, appeal), face-work (Goffman).
  - `subtext` (max 25 words): what the person does not say, avoids or reveals unintentionally; the incongruence between words and behaviour.
  - `bottom_line` (max 25 words): the brutally honest bottom line on this person's conduct here, the sentence a trusted adviser would say behind closed doors. No compliment sandwich; praise only when earned and specific.
  - `lever` (max 25 words): how to reach or handle this person in the next exchange, as a concrete move that changes behaviour, not appeasement ("Ask for a concrete number, then stay silent"; "Let them phrase the conclusion to restore status").
  - `credibility` (max 20 words): how well the statements are supported by concrete detail and consistency, with the reason.
- `dynamics` (2-4 sentences): the system between the speakers: who holds power and how (French and Raven: position, expertise, reward, coercion, likability); status moves (high and low status); alliances and who is left out; the game being played (Berne: "Yes, but", "wooden leg"); drama-triangle roles and how they switch; symmetric or complementary escalation (Watzlawick); the group's basic assumption (Bion: dependency, fight or flight, pairing); groupthink. For a monologue, the speaker's relation to the implied audience.
- `undercurrent` (1-2 sentences): what is in the room but nobody says: the unspoken conflict, the elephant, the real agenda behind the official one, the shared fear.
- `risks` (1-2 sentences): evasion, contradiction and defence patterns worth a closer look, and possible deception only when several flags agree. If none, say the speech appears straightforward.
- `recommendation` (max 25 words): the single most useful next move now: imperative, concrete, naming the decisive detail ("Next: pin the Friday risk to one named blocker before anyone agrees to a date."). If the conversation gives no basis yet, say so in one short sentence.
- `trend` (one sentence): how the picture changed against the previous assessment (calmer, more defensive, a mask slipped, new alliance, new topic). With no previous assessment, say this is the first.

## Rules
- The previous assessment serves continuity of wording and `trend` only. It is NOT evidence: drop what the transcript, flags and metrics now given no longer support. Never treat your own earlier wording as confirmation.
- Ground every statement in the material; quote short phrases when they carry the point. Metrics are heuristic indicators 0-100 (tension 1-10): translate them into plain language, never list raw numbers. Describe each speaker from their own metrics.
- Two speakers never get interchangeable profiles: find what is specific to each. Do not spread blame evenly when the material does not.
- A speaker flagged as having too little material (fewer than 25 words): write "insufficient material" in every deep field instead of inventing depth.
- Observable behaviour is stated as fact; an inference about inner motives is hedged once ("probably", "suggests"), then committed to. Weak verbal cues ("to be honest") are not deception.
- No psychiatric or medical diagnoses or disorder labels; describe the pattern. Never derive anything from gender, origin, age, accent or dialect.
- Never output internal codes (`defense`, `probe_question`); say it in words."""


def _score_prop(lo: int = 0, hi: int = 100) -> Dict[str, Any]:
    return {"type": "integer", "minimum": lo, "maximum": hi}


INSIGHT_SCHEMA: Dict[str, Any] = strict_schema({
        "type": "object",
        "properties": {
            "speaker_id": {"type": "string"},
            "evidence": {"type": "string", "description": "Max 25 words, 'cue -> reading'"},
            "cognitive_tension_score": _score_prop(1, 10),
            "mood": {"type": "string"},
            "valence": _score_prop(-100, 100),
            "arousal": _score_prop(),
            "dominance": _score_prop(),
            "stress": _score_prop(),
            "defensiveness": _score_prop(),
            "confidence": _score_prop(),
            "agitation": _score_prop(),
            "coherence": _score_prop(),
            "underlying_intent": {"type": "string"},
            "flags": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "phrase": {"type": "string", "description": "Verbatim from the NEW utterance"},
                        "category": {"type": "string", "enum": list(FLAG_CATEGORIES)},
                        "reason": {"type": "string"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["phrase", "category", "reason", "severity"],
                },
            },
            "ideas": {
                "type": "array",
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Verbatim from the NEW utterance"},
                        "kind": {"type": "string", "enum": ["approach", "task", "check", "wish"]},
                        "idea": {"type": "string"},
                        "why": {"type": "string"},
                        "novelty": _score_prop(),
                        "spin": {"type": "string"},
                    },
                    "required": ["source", "kind", "idea", "why", "novelty", "spin"],
                },
            },
            "intervention_type": {"type": "string", "enum": list(INTERVENTIONS)},
            "tactical_action": {"type": "string"},
        },
        "required": [
            "speaker_id", "evidence", "cognitive_tension_score", "mood", "valence", "arousal", "dominance",
            "stress", "defensiveness", "confidence", "agitation", "coherence", "underlying_intent",
            "flags", "ideas", "intervention_type", "tactical_action",
        ],
})

_PROFILE_FIELDS = ("role", "evidence", "state", "motives", "defenses", "relation", "subtext",
                   "bottom_line", "lever", "credibility")

SYNTH_SCHEMA: Dict[str, Any] = strict_schema({
        "type": "object",
        "properties": {
            "tells": {
                "type": "array",
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {"who": {"type": "string"}, "quote": {"type": "string"}, "pattern": {"type": "string"}},
                    "required": ["who", "quote", "pattern"],
                },
            },
            "headline": {"type": "string"},
            "situation": {"type": "string"},
            "speakers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}, **{f: {"type": "string"} for f in _PROFILE_FIELDS}},
                    "required": ["label", *_PROFILE_FIELDS],
                },
            },
            "dynamics": {"type": "string"},
            "undercurrent": {"type": "string"},
            "risks": {"type": "string"},
            "recommendation": {"type": "string"},
            "trend": {"type": "string"},
        },
        "required": ["tells", "headline", "situation", "speakers", "dynamics", "undercurrent",
                     "risks", "recommendation", "trend"],
})


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ASRPayload(BaseModel):
    utterance_id: str
    speaker: str
    text: str
    duration_ms: int = 0
    confidence: Optional[int] = Field(None, ge=0, le=100, description="Optional ASR confidence in percent")

    @field_validator("speaker", "text", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> str:
        return str(v).strip()

    @property
    def unreliable(self) -> bool:
        return self.confidence is not None and self.confidence < UNRELIABLE_BELOW


class Flag(BaseModel):
    phrase: str
    category: Literal["defense", "power", "deceptive", "evasive", "contradiction", "difficult", "emotional"]
    reason: str
    severity: Literal["low", "medium", "high"]


class Idea(BaseModel):
    source: str = ""
    kind: Literal["approach", "task", "check", "wish"] = "approach"
    idea: str
    why: str = ""
    novelty: int = Field(ge=0, le=100)
    spin: str = ""


class Insight(BaseModel):
    speaker_id: str
    evidence: str = ""
    cognitive_tension_score: int = Field(ge=1, le=10)
    mood: str
    valence: int = Field(0, ge=-100, le=100)
    arousal: int = Field(50, ge=0, le=100)
    dominance: int = Field(50, ge=0, le=100)
    stress: int = Field(50, ge=0, le=100)
    defensiveness: int = Field(50, ge=0, le=100)
    confidence: int = Field(50, ge=0, le=100)
    agitation: int = Field(50, ge=0, le=100)
    coherence: int = Field(50, ge=0, le=100)
    underlying_intent: str
    intervention_type: Literal["probe_question", "reframe", "acknowledge", "pause", "boundary", "redirect", "hold"] = "hold"
    tactical_action: str
    flags: List[Flag] = Field(default_factory=list)
    ideas: List[Idea] = Field(default_factory=list)


class Tell(BaseModel):
    who: str
    quote: str
    pattern: str


class SpeakerProfile(BaseModel):
    label: str
    role: str = ""
    evidence: str = ""
    state: str = ""
    motives: str = ""
    defenses: str = ""
    relation: str = ""
    subtext: str = ""
    bottom_line: str = ""
    lever: str = ""
    credibility: str = ""


class Synthesis(BaseModel):
    tells: List[Tell] = Field(default_factory=list)
    headline: str
    situation: str
    speakers: List[SpeakerProfile] = Field(default_factory=list)
    dynamics: str
    undercurrent: str = ""
    risks: str = ""
    recommendation: str
    trend: str = ""
    # filled by the engine, not by the model
    reliability: int = 0
    utterances_covered: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    ts: float = 0.0


@dataclass
class AnalysisResult:
    utterance_id: str
    speaker_id: str
    text: str
    insight: Optional[Insight]
    latency_s: float
    cost_usd: float
    total_spend_usd: float
    skipped_reason: str = ""
    ts: float = 0.0
    smoothed: Dict[str, float] = field(default_factory=dict)   # speaker's smoothed metrics after this utterance
    unreliable: bool = False


@dataclass
class SpeakerStats:
    n: int = 0
    words: int = 0
    unreliable: int = 0
    ema: Dict[str, float] = field(default_factory=dict)

    def update(self, ins: Insight) -> None:
        for k in SMOOTH_KEYS:
            x = float(getattr(ins, k))
            self.ema[k] = x if k not in self.ema else SMOOTH_ALPHA * x + (1 - SMOOTH_ALPHA) * self.ema[k]
        self.n += 1


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Persisted spend tracker. Refuses new calls once the projected spend crosses the cap."""

    def __init__(self, limit: float, path: Path) -> None:
        self.limit = limit
        self.path = path
        self.spent = 0.0
        try:
            self.spent = float(json.loads(path.read_text()).get("spent_usd", 0.0))
        except (OSError, ValueError, AttributeError):
            self.spent = 0.0

    def check(self, worst_case: float) -> None:
        if self.spent + worst_case >= self.limit:
            raise BudgetExceeded(f"Budget ${self.limit:.2f} reached (spent ${self.spent:.4f}).")

    def add_cost(self, cost: float) -> float:
        self.spent += cost
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"spent_usd": round(self.spent, 6), "updated": time.time()}))
            tmp.replace(self.path)
        except OSError:
            log.warning("Could not persist spend to %s", self.path)
        return cost


# --------------------------------------------------------------------------- #
# Verification helpers (the model's quotes are checked against the real text)
# --------------------------------------------------------------------------- #
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def norm_text(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.casefold()).split())


def _contains(haystack_norm: str, needle: str) -> bool:
    n = norm_text(needle)
    return len(n) >= 2 and n in haystack_norm


def _int(data: Dict[str, Any], key: str, lo: int, hi: int, default: Optional[int]) -> int:
    raw = data.get(key)
    if raw is None:
        if default is None:
            raise ValueError(f"missing {key}")
        return default
    return max(lo, min(hi, int(round(float(raw)))))


def verify_flags(raw: Any, text_norm: str) -> List[Flag]:
    out: List[Flag] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            flag = Flag(**item)
        except (ValidationError, TypeError):
            continue
        if not _contains(text_norm, flag.phrase):
            log.debug("dropped flag, phrase not in utterance: %r", flag.phrase)
            continue
        if flag.category == "deceptive" and flag.severity == "high":
            flag.severity = "medium"   # never high on wording alone
        out.append(flag)
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(out, key=lambda f: order[f.severity])[:3]


def clean_ideas(raw: Any, text_norm: str) -> List[Idea]:
    out: List[Idea] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            idea = Idea(**item)
        except (ValidationError, TypeError):
            continue
        if idea.kind != "approach" or idea.novelty < MIN_IDEA_NOVELTY or not _contains(text_norm, idea.source):
            continue
        out.append(idea)
    return out[:1]


class AgentCore:
    """
    Producer: submit(payload) from any coroutine on the same loop.
    Consumers: `history` (append-only, index-cursor friendly), `results` queue, `synthesis` / `synth_version`.

    Per-utterance calls are stateless: system prompt + rolling window (last WINDOW_SIZE utterances with their
    earlier reads) + the speaker's smoothed baseline + the new utterance. Every SYNTH_EVERY analysed utterances
    a separate, larger synthesis call profiles each speaker from the retained transcript; it runs as its own
    task so it never delays per-utterance analysis.
    """

    def __init__(self, provider: Any) -> None:
        self.provider = provider   # AnthropicProvider or GeminiProvider; owned (and closed) by the caller
        self.budget = Budget(BUDGET_USD, BUDGET_FILE)
        self.inbox: "asyncio.Queue[ASRPayload]" = asyncio.Queue()
        self.results: "asyncio.Queue[AnalysisResult]" = asyncio.Queue()
        self.history: List[AnalysisResult] = []
        self.window: Deque[Dict[str, Any]] = deque(maxlen=WINDOW_SIZE)
        self.transcript: Deque[Tuple[str, str, bool]] = deque(maxlen=TRANSCRIPT_KEEP)
        self.speaker_stats: Dict[str, SpeakerStats] = {}
        self.synthesis: Optional[Synthesis] = None
        self.synth_history: List[Synthesis] = []
        self.synth_version = 0
        self.synth_error = ""
        self.synth_running = False
        self.last_error: str = ""
        self._since_synth = 0
        self._task: Optional[asyncio.Task] = None
        self._synth_task: Optional[asyncio.Task] = None

    # ---- public ----------------------------------------------------------
    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="oxmoira-agent")

    async def stop(self) -> None:
        for t in (self._task, self._synth_task):
            if t:
                t.cancel()
        await asyncio.gather(*(t for t in (self._task, self._synth_task) if t), return_exceptions=True)
        self._task = self._synth_task = None

    def submit(self, raw: Dict[str, Any]) -> None:
        try:
            self.inbox.put_nowait(ASRPayload(**raw))
        except (ValidationError, TypeError) as exc:
            log.error("Bad ASR payload %r: %s", raw, exc)

    def request_synthesis(self) -> bool:
        """Manual trigger (e.g. a UI button). Returns False if one is already running or there is no material."""
        if self.synth_running or not self.transcript:
            return False
        self.synth_running = True
        self._synth_task = asyncio.create_task(self._synthesize(), name="oxmoira-synth")
        return True

    # ---- loop ------------------------------------------------------------
    async def _run(self) -> None:
        while True:
            payload = await self.inbox.get()
            # Backlog collapse: if analysis fell behind, absorb older utterances into the window
            # without paying for a call each, and analyze only the newest.
            while not self.inbox.empty():
                stale = payload
                self._remember(stale, None)
                self._emit(AnalysisResult(stale.utterance_id, stale.speaker, stale.text, None,
                                          0.0, 0.0, self.budget.spent, "backlog-collapsed", time.time(),
                                          unreliable=stale.unreliable))
                payload = self.inbox.get_nowait()
            try:
                result = await self._analyze(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep the loop alive no matter what
                log.exception("Analysis failed")
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._remember(payload, None)
                result = AnalysisResult(payload.utterance_id, payload.speaker, payload.text, None,
                                        0.0, 0.0, self.budget.spent, f"error: {self.last_error}", time.time(),
                                        unreliable=payload.unreliable)
            self._emit(result)
            if result.insight is not None and SYNTH_EVERY > 0:
                self._since_synth += 1
                if self._since_synth >= SYNTH_EVERY and not self.synth_running:
                    self.request_synthesis()

    def _emit(self, result: AnalysisResult) -> None:
        self.history.append(result)
        self.results.put_nowait(result)

    def _remember(self, p: ASRPayload, insight: Optional[Insight]) -> None:
        self.window.append({"speaker": p.speaker, "text": p.text, "insight": insight, "unreliable": p.unreliable,
                            "confidence": p.confidence})
        self.transcript.append((p.speaker, p.text, p.unreliable))
        st = self.speaker_stats.setdefault(p.speaker, SpeakerStats())
        st.words += len(p.text.split())
        if p.unreliable:
            st.unreliable += 1

    @staticmethod
    def _line(speaker: str, text: str, confidence: Optional[int], unreliable: bool) -> str:
        tag = f" (transcript confidence {confidence} %, UNRELIABLE)" if unreliable else ""
        return f"[{speaker}]{tag} {text}"

    def _render_window(self) -> str:
        lines: List[str] = []
        for e in self.window:
            line = self._line(e["speaker"], e["text"], e["confidence"], e["unreliable"])
            ins: Optional[Insight] = e["insight"]
            if ins is not None:
                line += f"  <read: tension={ins.cognitive_tension_score}, mood={ins.mood}>"
            lines.append(line)
        return "\n".join(lines) if lines else "(no prior utterances)"

    def _baseline(self, speaker: str) -> str:
        st = self.speaker_stats.get(speaker)
        if st is None or st.n < 2:
            return f"BASELINE for {speaker}: none yet ({0 if st is None else st.n} analysed utterances)."
        e = st.ema
        return (f"BASELINE for {speaker} ({st.n} analysed utterances, smoothed): tension {e['cognitive_tension_score']:.1f}/10, "
                f"stress {e['stress']:.0f}, defensiveness {e['defensiveness']:.0f}, agitation {e['agitation']:.0f}, "
                f"confidence {e['confidence']:.0f}, dominance {e['dominance']:.0f}, coherence {e['coherence']:.0f}.")

    # ---- tool-call plumbing -------------------------------------------------
    async def _json_call(self, system: str, schema: Dict[str, Any], user_msg: str, max_tokens: int,
                         effort: str, parse: Callable[[Dict[str, Any]], Any]) -> Tuple[Any, float, str]:
        total_cost = 0.0
        last_err = ""
        for _ in range(2):
            data, cost, err = await self.provider.json_call(system, schema, user_msg, max_tokens, effort)
            total_cost += cost
            self.budget.add_cost(cost)
            if data is None:
                last_err = err
                if "refusal" in err:
                    break
                continue
            try:
                return parse(data), total_cost, ""
            except (ValidationError, TypeError, ValueError) as exc:
                last_err = f"schema violation: {exc}"
        return None, total_cost, last_err

    def _worst_case(self, system: str, user_msg: str, out_tokens: int) -> float:
        est_in = (len(system) + len(user_msg)) // 3 + 500
        p_in, p_out = self.provider.price_in_out()
        return (est_in * p_in + out_tokens * p_out) / 1_000_000

    # ---- per-utterance analysis ---------------------------------------------
    async def _analyze(self, p: ASRPayload) -> AnalysisResult:
        if len(p.text.split()) < MIN_WORDS_TO_ANALYZE:
            self._remember(p, None)
            return AnalysisResult(p.utterance_id, p.speaker, p.text, None, 0.0, 0.0,
                                  self.budget.spent, "too-short", time.time(), unreliable=p.unreliable)

        user_msg = (
            f"WINDOW (oldest to newest):\n{self._render_window()}\n\n"
            f"{self._baseline(p.speaker)}\n\n"
            f"NEW UTTERANCE:\n{self._line(p.speaker, p.text, p.confidence, p.unreliable)}"
        )
        self.budget.check(self._worst_case(SYSTEM_PROMPT, user_msg, MAX_OUTPUT_TOKENS))
        text_norm = norm_text(p.text)

        def parse(data: Dict[str, Any]) -> Insight:
            flags_raw = data.pop("flags", [])
            ideas_raw = data.pop("ideas", [])
            data["cognitive_tension_score"] = _int(data, "cognitive_tension_score", 1, 10, None)
            data["valence"] = _int(data, "valence", -100, 100, 0)
            for k in SCORE_KEYS[1:]:
                data[k] = _int(data, k, 0, 100, 50)
            if p.unreliable:   # recognition errors are not incoherence
                data["coherence"] = max(45, min(60, data["coherence"]))
            data["speaker_id"] = p.speaker   # ASR label is authoritative
            if data.get("intervention_type") not in INTERVENTIONS:
                data["intervention_type"] = "hold"
            ins = Insight(**{k: v for k, v in data.items() if k in Insight.model_fields})
            ins.flags = verify_flags(flags_raw, text_norm)
            ins.ideas = clean_ideas(ideas_raw, text_norm)
            return ins

        t0 = time.monotonic()
        insight, cost, err = await self._json_call(SYSTEM_PROMPT, INSIGHT_SCHEMA, user_msg, MAX_OUTPUT_TOKENS, ANALYSIS_EFFORT, parse)
        latency = time.monotonic() - t0

        self._remember(p, insight)
        smoothed: Dict[str, float] = {}
        if insight is not None:
            st = self.speaker_stats[p.speaker]
            st.update(insight)
            smoothed = dict(st.ema)
            return AnalysisResult(p.utterance_id, p.speaker, p.text, insight, latency, cost,
                                  self.budget.spent, "", time.time(), smoothed, p.unreliable)
        log.error("Insight unavailable for %s: %s", p.utterance_id, err)
        return AnalysisResult(p.utterance_id, p.speaker, p.text, None, latency, cost,
                              self.budget.spent, f"parse-failed: {err}", time.time(), unreliable=p.unreliable)

    # ---- periodic synthesis -------------------------------------------------
    def _synth_input(self) -> Tuple[str, List[str], int]:
        lines = [f"[{s}]{' (UNRELIABLE)' if u else ''} {t}" for s, t, u in self.transcript]
        transcript = "\n".join(lines)
        if len(transcript) > SYNTH_MAX_CHARS:
            transcript = "(earlier lines omitted)\n" + transcript[-SYNTH_MAX_CHARS:].split("\n", 1)[-1]

        roster_lines: List[str] = []
        labels = list(dict.fromkeys(s for s, _, _ in self.transcript))
        total_words = 0
        for lab in labels:
            st = self.speaker_stats.get(lab, SpeakerStats())
            total_words += st.words
            thin = "  ** TOO LITTLE MATERIAL: write 'insufficient material' in every deep field **" if st.words < MIN_PROFILE_WORDS else ""
            e = st.ema
            metrics = (f"tension {e['cognitive_tension_score']:.1f}/10, stress {e['stress']:.0f}, defensiveness {e['defensiveness']:.0f}, "
                       f"agitation {e['agitation']:.0f}, confidence {e['confidence']:.0f}, dominance {e['dominance']:.0f}, "
                       f"coherence {e['coherence']:.0f}") if e else "no metrics yet"
            unrel = f", {st.unreliable} unreliable lines" if st.unreliable else ""
            roster_lines.append(f"- {lab}: {st.words} words{unrel}; smoothed metrics: {metrics}{thin}")

        flags: List[str] = []
        ideas: List[Tuple[int, str]] = []
        for r in self.history:
            if r.insight is None:
                continue
            for f in r.insight.flags:
                flags.append(f"[{r.speaker_id}] {f.category}/{f.severity}: \"{f.phrase}\" ({f.reason})")
            for i in r.insight.ideas:
                ideas.append((i.novelty, f"[{r.speaker_id}] \"{i.source}\" -> {i.idea} (novelty {i.novelty})"))
        ideas.sort(key=lambda x: -x[0])

        prev = "(none: this is the first assessment)"
        if self.synthesis is not None:
            prev = (f"headline: {self.synthesis.headline}\nsituation: {self.synthesis.situation}\n"
                    f"recommendation: {self.synthesis.recommendation}")
        user_msg = (
            f"SPEAKER ROSTER:\n{chr(10).join(roster_lines)}\n\n"
            f"LATEST FLAGS (engine-verified):\n{chr(10).join(flags[-15:]) or '(none)'}\n\n"
            f"MOST NOVEL IDEAS:\n{chr(10).join(t for _, t in ideas[:3]) or '(none)'}\n\n"
            f"PREVIOUS ASSESSMENT (continuity and trend only, NOT evidence):\n{prev}\n\n"
            f"TRANSCRIPT:\n<<<\n{transcript}\n>>>\n\n"
            "Return the overall assessment now."
        )
        return user_msg, labels, total_words

    async def _synthesize(self) -> None:
        self.synth_running = True
        self._since_synth = 0
        try:
            user_msg, labels, total_words = self._synth_input()
            self.budget.check(self._worst_case(SYNTH_SYSTEM_PROMPT, user_msg, SYNTH_OUTPUT_TOKENS))
            transcript_norm = norm_text("\n".join(t for _, t, _ in self.transcript))
            covered = len(self.transcript)

            def parse(data: Dict[str, Any]) -> Synthesis:
                tells: List[Tell] = []
                for item in data.pop("tells", []) or []:
                    try:
                        tell = Tell(**item)
                    except (ValidationError, TypeError):
                        continue
                    if _contains(transcript_norm, tell.quote):
                        tells.append(tell)
                profiles: List[SpeakerProfile] = []
                for item in data.pop("speakers", []) or []:
                    try:
                        prof = SpeakerProfile(**item)
                    except (ValidationError, TypeError):
                        continue
                    if prof.label not in labels:
                        continue
                    st = self.speaker_stats.get(prof.label)
                    if st is not None and st.words < MIN_PROFILE_WORDS:   # enforced in code, not only in the prompt
                        for f in _PROFILE_FIELDS[1:]:
                            setattr(prof, f, "insufficient material")
                    profiles.append(prof)
                syn = Synthesis(tells=tells[:10], speakers=profiles,
                                **{k: v for k, v in data.items() if k in Synthesis.model_fields and k not in ("tells", "speakers")})
                return syn

            t0 = time.monotonic()
            syn, cost, err = await self._json_call(SYNTH_SYSTEM_PROMPT, SYNTH_SCHEMA, user_msg, SYNTH_OUTPUT_TOKENS, SYNTH_EFFORT, parse)
            if syn is None:
                self.synth_error = err
                log.error("Synthesis failed: %s", err)
                return
            unreliable_frac = sum(1 for _, _, u in self.transcript if u) / max(1, covered)
            syn.reliability = max(0, min(100, round(100 * min(1.0, total_words / 200) * (1 - 0.5 * unreliable_frac))))
            syn.utterances_covered = covered
            syn.latency_s = time.monotonic() - t0
            syn.cost_usd = cost
            syn.ts = time.time()
            self.synthesis = syn
            self.synth_history.append(syn)
            self.synth_version += 1
            self.synth_error = ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.synth_error = f"{type(exc).__name__}: {exc}"
            log.exception("Synthesis crashed")
        finally:
            self.synth_running = False


async def _selftest() -> None:
    from providers import AnthropicProvider, DEFAULT_ANTHROPIC_MODEL
    logging.basicConfig(level=logging.INFO)
    provider = AnthropicProvider(os.environ["ANTHROPIC_API_KEY"], os.getenv("OXMOIRA_MODEL", DEFAULT_ANTHROPIC_MODEL))
    agent = AgentCore(provider)
    agent.start()
    samples = [
        ("Speaker 1", "We cannot deploy this pipeline by Friday, the risk is too high."),
        ("Speaker 2", "I hear you, but the client was promised Friday and I already told them yes. Honestly, it's not that big a deal."),
    ]
    for i, (spk, text) in enumerate(samples):
        agent.submit({"utterance_id": f"chunk_{i}", "speaker": spk, "text": text, "duration_ms": 3000})
        r = await agent.results.get()
        print(json.dumps(r.insight.model_dump() if r.insight else {"skipped": r.skipped_reason}, indent=2))
        print(f"latency={r.latency_s:.2f}s cost=${r.cost_usd:.5f} total=${r.total_spend_usd:.5f}")
    if agent.request_synthesis():
        while agent.synth_running:
            await asyncio.sleep(0.2)
        print(agent.synthesis.model_dump_json(indent=2) if agent.synthesis else f"synthesis failed: {agent.synth_error}")
    await agent.stop()
    await provider.close()


if __name__ == "__main__":
    asyncio.run(_selftest())
