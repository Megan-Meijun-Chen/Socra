"""
Socra – socra_core.py
=====================
Multi-Agent Debate engine. Zero UI dependencies.

Two AI models independently answer a question (Round 0, parallel),
then swap answers and critique each other each round until both agree
or max_rounds is reached.

on_update(event, data) callback fires after every significant step so
the UI can display results in real time.
"""

import os, json, re, pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ── Load .env: local first, then sibling quorum folder ─────
_here = pathlib.Path(__file__).parent
load_dotenv(_here / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    load_dotenv(_here.parent / "quorum" / ".env")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY",    "")
GOOGLE_API_KEY    = os.environ.get("GOOGLE_API_KEY",    "")

MODEL_COSTS = {
    "Claude": {"input": 0.015,   "output": 0.075},
    "GPT-4":  {"input": 0.0025,  "output": 0.010},
    "Gemini": {"input": 0.00125, "output": 0.005},
}

# ═══════════════════════════════════════════════════════════
# API status
# ═══════════════════════════════════════════════════════════

def get_api_status() -> dict:
    return {
        "Claude": bool(ANTHROPIC_API_KEY),
        "GPT-4":  bool(OPENAI_API_KEY),
        "Gemini": bool(GOOGLE_API_KEY),
    }

# ═══════════════════════════════════════════════════════════
# System prompts
# ═══════════════════════════════════════════════════════════

_ROUND0_BASE = """\
You are a precise, analytical assistant. Answer the question as accurately as possible.
- Math / logic: show step-by-step reasoning
- Factual questions: state what you know and your uncertainty
- Open questions: provide a well-reasoned, structured answer
{lang_instruction}
Output ONLY valid JSON — no markdown wrapping:
{{
  "answer":     "your complete answer",
  "reasoning":  "step-by-step reasoning or key supporting points",
  "confidence": <integer 0-100>
}}"""

_DEBATE_BASE = """\
You are in a verification debate. You previously answered a question.
Now you see another AI's answer. Your job: evaluate it rigorously.

Rules:
- If you find an error in their answer, explain exactly what is wrong and why
- If you find an error in YOUR OWN previous answer, acknowledge it and correct yourself
- If both answers are essentially correct, confirm this clearly
- Be direct — accuracy matters more than politeness
- agreement_score: 0 = completely different answers, 100 = fully identical answers
- key_disagreement: the single most important point of disagreement (null if you agree)
{lang_instruction}
Output ONLY valid JSON — no markdown wrapping:
{{
  "agrees":           <true or false>,
  "agreement_score":  <integer 0-100>,
  "key_disagreement": "the specific point you disagree on, or null if you agree",
  "critique":         "specific evaluation — what is right, what is wrong, and why",
  "refined_answer":   "your current best answer (updated if you found errors, same if still correct)"
}}"""

_SYNTHESIS_BASE = """\
You are summarizing a multi-AI verification debate concisely.
{lang_instruction}
Output valid JSON only — no markdown wrapping:
{{
  "final_answer":        "the best answer based on the full debate, clear and complete",
  "verification_summary":"1-2 sentences describing what happened in the debate",
  "key_disagreement":    "the main point they disagreed on during debate, or null if they agreed throughout"
}}"""

_FOLLOWUP_BASE = """\
You are helping a student who just received a verified answer from a multi-AI debate.
Below is the original question and a summary of the debate result.
Answer their follow-up question clearly and helpfully.
{lang_instruction}
Output ONLY valid JSON — no markdown wrapping:
{{
  "answer": "your complete response to the follow-up question"
}}"""

_LANG_LINES = {
    "zh": "IMPORTANT: You must write ALL your responses entirely in Simplified Chinese (简体中文). Do not use English.",
    "en": "IMPORTANT: You must write ALL your responses entirely in English.",
}

def _build_systems(lang: str = "zh") -> tuple[str, str, str, str]:
    """Return (ROUND0_SYSTEM, DEBATE_SYSTEM, SYNTHESIS_SYSTEM, FOLLOWUP_SYSTEM) with lang baked in."""
    line = _LANG_LINES.get(lang, _LANG_LINES["zh"])
    return (
        _ROUND0_BASE.format(lang_instruction=line),
        _DEBATE_BASE.format(lang_instruction=line),
        _SYNTHESIS_BASE.format(lang_instruction=line),
        _FOLLOWUP_BASE.format(lang_instruction=line),
    )

# ═══════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text) // 4

def calc_cost(model: str, in_tok: int, out_tok: int) -> float:
    r = MODEL_COSTS.get(model, {"input": 0, "output": 0})
    return in_tok / 1000 * r["input"] + out_tok / 1000 * r["output"]

def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"answer": raw[:500], "reasoning": "", "confidence": 50, "_parse_error": True}

def _retry():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )

# ═══════════════════════════════════════════════════════════
# Model callers
# ═══════════════════════════════════════════════════════════

@_retry()
def _call_claude(system: str, prompt: str, image_b64: str = None, image_mime: str = "image/jpeg") -> tuple[str, int, int]:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    if image_b64:
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": image_mime, "data": image_b64}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1200,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return msg.content[0].text, msg.usage.input_tokens, msg.usage.output_tokens

@_retry()
def _call_gpt4(system: str, prompt: str, image_b64: str = None, image_mime: str = "image/jpeg") -> tuple[str, int, int]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    if image_b64:
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = prompt
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1200,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
    )
    return (
        resp.choices[0].message.content,
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )

@_retry()
def _call_gemini(system: str, prompt: str, image_b64: str = None, image_mime: str = "image/jpeg") -> tuple[str, int, int]:
    import google.generativeai as genai
    import base64 as _b64
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=system,
        generation_config={"response_mime_type": "application/json"},
    )
    if image_b64:
        image_part = {"mime_type": image_mime, "data": _b64.b64decode(image_b64)}
        contents = [image_part, prompt]
    else:
        contents = prompt
    resp    = model.generate_content(contents)
    raw     = resp.text
    in_tok  = count_tokens(prompt)
    out_tok = count_tokens(raw)
    return raw, in_tok, out_tok

_CALLERS = {
    "Claude": _call_claude,
    "GPT-4":  _call_gpt4,
    "Gemini": _call_gemini,
}

def call_model(model: str, system: str, prompt: str, image_b64: str = None, image_mime: str = "image/jpeg") -> tuple[dict, int, int, float]:
    raw, in_tok, out_tok = _CALLERS[model](system, prompt, image_b64, image_mime)
    return parse_json(raw), in_tok, out_tok, calc_cost(model, in_tok, out_tok)

# ═══════════════════════════════════════════════════════════
# Debate engine
# ═══════════════════════════════════════════════════════════

def run_debate(
    question:   str,
    model_a:    str,
    model_b:    str,
    max_rounds: int = 3,
    lang:       str = "zh",
    on_update:  Optional[Callable] = None,
    image_b64:  Optional[str] = None,
    image_mime: str = "image/jpeg",
) -> dict:
    """
    Run multi-round debate between two models.

    on_update(event, data) events:
      "round_complete"  → data = round dict
      "converged"       → data = {"round": int}
      "max_rounds"      → data = {}

    Returns:
      {converged, converged_at_round, final_answer, verification_summary,
       key_disagreement, debate_log, total_cost, model_a, model_b}
    """
    debate_log  = []
    total_cost  = 0.0
    prompt_r0   = f"Question: {question}"
    R0_SYS, DEBATE_SYS, SYNTH_SYS, _ = _build_systems(lang)

    # ── Round 0: both models answer independently (parallel) ──
    def _answer(model):
        try:
            parsed, it, ot, cost = call_model(model, R0_SYS, prompt_r0, image_b64, image_mime)
            return {"ok": True, "data": parsed, "cost": cost}
        except Exception as e:
            return {"ok": False, "error": str(e), "cost": 0.0}

    with ThreadPoolExecutor(max_workers=2) as pool:
        fa, fb  = pool.submit(_answer, model_a), pool.submit(_answer, model_b)
        ra, rb  = fa.result(), fb.result()

    total_cost += ra["cost"] + rb["cost"]
    round0 = {
        "round": 0, "type": "independent",
        "responses": {model_a: ra, model_b: rb},
    }
    debate_log.append(round0)
    if on_update:
        on_update("round_complete", round0)

    if not ra.get("ok") or not rb.get("ok"):
        return {
            "converged": False, "error": True,
            "debate_log": debate_log, "total_cost": total_cost,
            "model_a": model_a, "model_b": model_b,
        }

    # Track each model's latest answer for prompting
    prev = {model_a: ra["data"], model_b: rb["data"]}

    # ── Rounds 1+: cross-examination (parallel) ───────────
    for rnd in range(1, max_rounds + 1):

        # Capture current prev snapshot for thread safety
        snap_a = json.dumps(prev[model_a], ensure_ascii=False)
        snap_b = json.dumps(prev[model_b], ensure_ascii=False)

        def _cross(model, my_snap, their_snap):
            prompt = (
                f"Original question: {question}\n\n"
                f"Your previous answer:\n{my_snap}\n\n"
                f"The other AI's answer:\n{their_snap}\n\n"
                f"Evaluate and respond in the required JSON format."
            )
            try:
                parsed, it, ot, cost = call_model(model, DEBATE_SYS, prompt, image_b64, image_mime)
                return {"ok": True, "data": parsed, "cost": cost}
            except Exception as e:
                return {"ok": False, "error": str(e), "cost": 0.0}

        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(_cross, model_a, snap_a, snap_b)
            fb = pool.submit(_cross, model_b, snap_b, snap_a)
            ca, cb = fa.result(), fb.result()

        total_cost += ca["cost"] + cb["cost"]
        round_data = {
            "round": rnd, "type": "debate",
            "responses": {model_a: ca, model_b: cb},
        }
        debate_log.append(round_data)
        if on_update:
            on_update("round_complete", round_data)

        # Update prev for next round
        if ca.get("ok"):
            prev[model_a] = ca["data"]
        if cb.get("ok"):
            prev[model_b] = cb["data"]

        # Check convergence: both models explicitly agree
        a_agrees = ca.get("ok") and ca["data"].get("agrees", False)
        b_agrees = cb.get("ok") and cb["data"].get("agrees", False)

        if a_agrees and b_agrees:
            if on_update:
                on_update("converged", {"round": rnd})
            final, summary, key_dis, sc = _synthesize(
                question, debate_log, model_a, model_b,
                converged=True, synth_sys=SYNTH_SYS, synth_model=model_a,
            )
            return {
                "converged": True,
                "converged_at_round": rnd,
                "final_answer": final,
                "verification_summary": summary,
                "key_disagreement": key_dis,
                "debate_log": debate_log,
                "total_cost": total_cost + sc,
                "model_a": model_a,
                "model_b": model_b,
            }

    # Max rounds reached without full convergence
    if on_update:
        on_update("max_rounds", {})
    final, summary, key_dis, sc = _synthesize(
        question, debate_log, model_a, model_b,
        converged=False, synth_sys=SYNTH_SYS, synth_model=model_a,
    )
    return {
        "converged": False,
        "converged_at_round": None,
        "final_answer": final,
        "verification_summary": summary,
        "key_disagreement": key_dis,
        "debate_log": debate_log,
        "total_cost": total_cost + sc,
        "model_a": model_a,
        "model_b": model_b,
    }


def _synthesize(
    question:    str,
    debate_log:  list,
    model_a:     str,
    model_b:     str,
    converged:   bool,
    synth_sys:   str = "",
    synth_model: str = "Claude",
) -> tuple[str, str, str, float]:
    """
    Generate final answer + verification summary using one of the chosen models.
    Returns (final_answer, verification_summary, key_disagreement, cost).
    synth_model defaults to model_a — no longer hard-coded to Claude.
    """
    lines = [f"Question: {question}\n"]
    for entry in debate_log:
        lines.append(f"Round {entry['round']} ({entry['type']}):")
        for model, res in entry["responses"].items():
            if res.get("ok"):
                d = res["data"]
                if entry["round"] == 0:
                    lines.append(f"  {model}: {d.get('answer', '')}")
                else:
                    agree_str = "AGREES" if d.get("agrees") else "DISAGREES"
                    score     = d.get("agreement_score", "?")
                    lines.append(f"  {model}: {agree_str} (score {score}) — {d.get('critique', '')}")
                    if d.get("key_disagreement"):
                        lines.append(f"    Key disagreement: {d['key_disagreement']}")
                    if d.get("refined_answer"):
                        lines.append(f"    → Refined: {d['refined_answer']}")
        lines.append("")

    status_line = (
        "The two models reached consensus."
        if converged else
        "The models did not fully converge after the maximum rounds."
    )
    meta = (
        "\n".join(lines)
        + f"\nStatus: {status_line}\n\n"
        + "Provide the required JSON with final_answer, verification_summary, and key_disagreement."
    )

    # Use whichever model was chosen — no more Claude-only synthesis
    sys = synth_sys or _build_systems("zh")[2]

    # Fallback chain: try synth_model first, then any available model
    fallback_order = [synth_model, model_a, model_b, "Claude", "GPT-4", "Gemini"]
    seen = set()
    for m in fallback_order:
        if m in seen or m not in _CALLERS:
            continue
        seen.add(m)
        api_keys = {"Claude": ANTHROPIC_API_KEY, "GPT-4": OPENAI_API_KEY, "Gemini": GOOGLE_API_KEY}
        if not api_keys.get(m, ""):
            continue
        try:
            parsed, it, ot, cost = call_model(m, sys, meta)
            return (
                parsed.get("final_answer", ""),
                parsed.get("verification_summary", ""),
                parsed.get("key_disagreement") or "",
                cost,
            )
        except Exception:
            continue

    # Hard fallback: use last refined answer
    last = debate_log[-1]["responses"].get(model_a, {})
    if last.get("ok"):
        d   = last["data"]
        ans = d.get("refined_answer") or d.get("answer", "")
        return ans, "Synthesis unavailable.", "", 0.0
    return "", "Synthesis unavailable.", "", 0.0


# ═══════════════════════════════════════════════════════════
# Follow-up conversation
# ═══════════════════════════════════════════════════════════

def run_followup(
    original_question: str,
    debate_result:     dict,
    followup_question: str,
    model:             str,
    lang:              str = "zh",
) -> dict:
    """
    Answer a follow-up question using one model, with the full debate as context.
    Returns {"answer": str, "cost": float, "ok": bool}.
    """
    _, _, _, FOLLOWUP_SYS = _build_systems(lang)

    # Build context from the debate
    final   = debate_result.get("final_answer", "")
    summary = debate_result.get("verification_summary", "")
    model_a = debate_result.get("model_a", "")
    model_b = debate_result.get("model_b", "")

    prompt = (
        f"Original question: {original_question}\n\n"
        f"Verified answer from debate between {model_a} and {model_b}:\n{final}\n\n"
        f"Debate summary: {summary}\n\n"
        f"Student follow-up question: {followup_question}"
    )

    try:
        parsed, it, ot, cost = call_model(model, FOLLOWUP_SYS, prompt)
        return {"ok": True, "answer": parsed.get("answer", ""), "cost": cost}
    except Exception as e:
        return {"ok": False, "answer": str(e), "cost": 0.0}
