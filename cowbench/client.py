"""Talking to the vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import hashlib
import json
import time

import requests

from cbvd import ACTIVITIES, POSTURES

# Behaviour wording follows the CBVD-5 paper's definitions, not the labelmap
# strings: the model is asked what it can see, and "foraging" / "drinking
# water" / "rumination" are dataset vocabulary, not a description of the scene.
_HEADER_SINGLE = """You are looking at a single still frame from a fixed surveillance camera in a dairy barn.

Exactly one cow is outlined with a bright green rectangle. Judge only that cow. Ignore every other animal in the frame."""

_HEADER_MULTI = """You are looking at {n} frames from a fixed surveillance camera in a dairy barn, in order, spanning {span:g} seconds of the same scene.

The same cow is outlined with a bright green rectangle in every frame. Judge only that cow. Ignore every other animal. Use the movement between frames, not just the last one.

The outline was drawn on the middle frame, so a cow that moves may sit slightly off its rectangle in the first and last frames; follow the animal, not the box."""

_BODY = """

Report two things about the outlined cow.

posture - exactly one of:
  standing  the cow carries its weight on its legs, body upright
  lying     the cow's body rests on the stall bed or the floor

activity - exactly one of:
  feeding     the head is down at the feed barrier or in the feed alley, eating or reaching for feed
  drinking    the head is over or inside a water trough
  ruminating  chewing cud while not at feed and not at water; the jaw works sideways, head up or resting
  none        none of the three above can be seen

Both fields are independent: a lying cow can be ruminating, a standing cow can be feeding.

Answer with JSON only."""


def build_prompt(n_frames: int = 1, span: float = 0.0) -> str:
    header = (_HEADER_SINGLE if n_frames <= 1
              else _HEADER_MULTI.format(n=n_frames, span=span))
    return header + _BODY


def prompt_sha(n_frames: int = 1, span: float = 0.0) -> str:
    """Goes in the report: two runs whose numbers are compared must have been
    asked the same question, and this is what proves it."""
    return hashlib.sha256(
        build_prompt(n_frames, span).encode("utf-8")).hexdigest()[:12]


PROMPT = build_prompt()
PROMPT_SHA = prompt_sha()

# Structured output, not free text plus a regex. A parser is one more thing
# that can be wrong, and a parse failure is indistinguishable in the numbers
# from the model actually being wrong.
SCHEMA = {
    "type": "object",
    "properties": {
        "posture": {"type": "string", "enum": list(POSTURES)},
        "activity": {"type": "string", "enum": list(ACTIVITIES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["posture", "activity", "confidence"],
    "additionalProperties": False,
}


# --answer-now: the assistant turn is started for the model, so it answers at
# once instead of reasoning first - what a LoRA from lora/train_lora.py was
# trained on, token for token: the chat template renders this prefill as
# "<|start|>assistant to=user<|message|>{"posture": "". No json_schema then:
# guided decoding would demand a fresh object, not the rest of this one, and
# the adapter answers without the confidence field the schema requires.
ANSWER_PREFILL = '{"posture": "'


def parse_enums(text: str):
    """The two fields from an answer that is not clean JSON: the enum values
    named after their keys."""
    import re
    out = {}
    for key, values in (("posture", POSTURES), ("activity", ACTIVITIES)):
        m = re.search(r'"%s"\s*:\s*"(%s)"' % (key, "|".join(values)), text)
        if m:
            out[key] = m.group(1)
    return out


class ModelError(RuntimeError):
    pass


class MuseClient:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 temperature: float = 0.0, max_tokens: int = 2048,
                 timeout: float = 300.0, retries: int = 3, answer_now: bool = False):
        self.answer_now = answer_now
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    def server_info(self) -> dict:
        info = {"base_url": self.base_url}
        try:
            models = self.session.get(f"{self.base_url}/v1/models", timeout=30).json()
            entry = next((m for m in models.get("data", []) if m.get("id") == self.model),
                         (models.get("data") or [None])[0])
            if entry:
                info["served_model_id"] = entry.get("id")
                info["model_root"] = entry.get("root")
                info["max_model_len"] = entry.get("max_model_len")
                # A LoRA's root is the adapter's path; the checkpoint it runs
                # on is its parent's root.
                parent = next((m for m in models.get("data", [])
                               if entry.get("parent") and m.get("id") == entry.get("parent")), None)
                if parent and parent is not entry:
                    info["adapter_root"] = entry.get("root")
                    info["base_model_root"] = parent.get("root")
        except Exception as exc:  # the run is still valid without provenance
            info["models_endpoint_error"] = str(exc)
        try:
            info["vllm_version"] = self.session.get(
                f"{self.base_url}/version", timeout=30).json().get("version")
        except Exception:
            pass
        return info

    def classify(self, data_urls, n_frames: int = 1, span: float = 0.0) -> dict:
        if self.answer_now:
            # About 12 tokens of answer; nothing to think through first.
            return self._once(data_urls, n_frames, span, 64)
        out = self._once(data_urls, n_frames, span, self.max_tokens)
        # Muse Glimmer thinks before it answers, and the more frames it is given
        # the longer it thinks: five frames routinely overrun a budget one frame
        # never touches. A run truncated mid-thought has no answer at all, and
        # counting that as "the model was wrong" would be a lie about the model.
        if out.get("finish_reason") == "length" and out.get("posture") is None:
            out = self._once(data_urls, n_frames, span, self.max_tokens * 2)
            out["retried_for_length"] = True
        return out

    def _post(self, payload) -> dict:
        last = None
        for attempt in range(self.retries):
            try:
                resp = self.session.post(f"{self.base_url}/v1/chat/completions",
                                         json=payload, timeout=self.timeout)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise ModelError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last = exc
                if attempt == self.retries - 1:
                    raise ModelError(str(exc)) from exc
                time.sleep(2 ** attempt)
        raise ModelError(str(last))  # pragma: no cover

    def ask(self, data_urls, text: str, schema: dict, name: str, max_tokens=None) -> dict:
        """One images+text question with a JSON-schema answer, for tasks other
        than the per-cow classification (detect.py). The parsed answer is in
        "parsed"; the rest is kept as classify keeps it."""
        content = [{"type": "image_url", "image_url": {"url": u}} for u in data_urls]
        content.append({"type": "text", "text": text})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "seed": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
        }
        t0 = time.perf_counter()
        body = self._post(payload)
        message = body["choices"][0]["message"]
        out = {
            "raw": message.get("content") or "",
            "reasoning": message.get("reasoning") or message.get("reasoning_content"),
            "finish_reason": body["choices"][0].get("finish_reason"),
            "usage": body.get("usage"),
            "seconds": round(time.perf_counter() - t0, 3),
        }
        try:
            out["parsed"] = json.loads(out["raw"])
        except Exception as exc:
            out["parse_error"] = f"{type(exc).__name__}: {exc}"
        return out

    def complete(self, data_urls, text, prefill=None, schema=None, name="answer",
                 max_tokens=None) -> dict:
        """images + text, answered either with the assistant turn started as
        `prefill` (no reasoning, no schema - how a LoRA from lora/train_lora.py
        answers) or under `schema` (reasoning on). "raw" is the full answer,
        prefill included. Used by the frame unit (frame.py)."""
        content = [{"type": "image_url", "image_url": {"url": u}} for u in data_urls]
        content.append({"type": "text", "text": text})
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "temperature": self.temperature, "max_tokens": max_tokens or self.max_tokens,
                   "seed": 0}
        if prefill is not None:
            payload["messages"].append({"role": "assistant", "content": prefill})
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        elif schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": name, "schema": schema, "strict": True}}
        t0 = time.perf_counter()
        body = self._post(payload)
        message = body["choices"][0]["message"]
        text_out = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if prefill is not None:
            # As in _once: a reasoning parser may file a prefilled answer under reasoning.
            text_out, reasoning = prefill + (text_out or reasoning or ""), None
        return {"raw": text_out, "reasoning": reasoning,
                "finish_reason": body["choices"][0].get("finish_reason"),
                "usage": body.get("usage"), "seconds": round(time.perf_counter() - t0, 3)}

    def _once(self, data_urls, n_frames, span, max_tokens) -> dict:
        content = [{"type": "image_url", "image_url": {"url": u}} for u in data_urls]
        content.append({"type": "text", "text": build_prompt(n_frames, span)})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "seed": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "cow_behaviour", "schema": SCHEMA, "strict": True},
            },
        }
        if self.answer_now:
            del payload["response_format"]
            payload["messages"].append({"role": "assistant", "content": ANSWER_PREFILL})
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False

        t0 = time.perf_counter()
        body = self._post(payload)
        seconds = time.perf_counter() - t0

        message = body["choices"][0]["message"]
        raw = message.get("content") or ""
        if self.answer_now:
            # A reasoning parser that finds no reasoning block may file the
            # whole answer under "reasoning"; the answer is wherever the text is.
            raw = ANSWER_PREFILL + (raw or message.get("reasoning")
                                    or message.get("reasoning_content") or "")
        out = {
            "raw": raw,
            # Muse Glimmer runs with --reasoning-parser, so the chain of thought
            # arrives in its own field - "reasoning" on vLLM 0.30, not the
            # "reasoning_content" older versions used. Kept for error analysis;
            # never parsed. It is also where the token budget goes: the model
            # spends a few hundred tokens here before emitting the JSON, which
            # is why max_tokens is 2048 and not 64.
            "reasoning": message.get("reasoning") or message.get("reasoning_content"),
            "finish_reason": body["choices"][0].get("finish_reason"),
            "usage": body.get("usage"),
            "seconds": round(seconds, 3),
        }
        try:
            parsed = json.loads(raw)
            out["posture"] = parsed.get("posture")
            out["activity"] = parsed.get("activity")
            out["confidence"] = parsed.get("confidence")
        except Exception as exc:
            found = parse_enums(raw) if self.answer_now else {}
            if found:
                out.update(found)
                out["parse_note"] = f"not clean JSON ({type(exc).__name__}); fields read by name"
            else:
                out["parse_error"] = f"{type(exc).__name__}: {exc}"
        return out
