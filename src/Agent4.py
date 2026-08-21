# ════════════════════════════════════════════════════════
# SECTION 9 — AGENT 4: INPUT CORRECTION (LLM / ReAct)
# ────────────────────────────────────────────────────────
# Responsibilities:
#   - Accept raw user query (may have typos/OCR errors)
#   - Send to Groq LLM (Llama 3.3) with domain context
#   - LLM corrects typos, spacing, capitalization
#   - Self-corrects if LLM response is malformed (retry)
#   - Returns corrected query + reason + confidence
# ════════════════════════════════════════════════════════

def _extract_json(raw: str) -> dict:
    """Robustly pull a JSON object out of an LLM response."""
    if not raw:
        raise ValueError("empty response")
    raw = raw.strip()
    # Strip markdown code fences if present.
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    # Grab the first {...} block (handles any stray prose around it).
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


# Focused, OCR-specific correction prompt. Tight rules keep the model from
# rewriting or hallucinating content — it only repairs scanning artifacts.
CORRECTION_SYSTEM_PROMPT = """You are an OCR post-correction engine for scanned documents.
You receive raw text extracted by Tesseract from a single document region.
Your ONLY job is to repair OCR scanning errors — not to rewrite the text.

FIX these common OCR mistakes:
- Character confusions: rn↔m, cl↔d, 0↔O, 1↔l↔I, 5↔S, 8↔B, vv↔w, |↔I.
- Words wrongly split ("inform ation" → "information") or merged ("ofthe" → "of the").
- Stray punctuation/symbols inserted mid-word, and broken spacing around punctuation.
- Obvious misspellings that are clearly scan artifacts.

STRICT RULES:
- Do NOT add, remove, summarize, translate, or reorder words.
- Do NOT change correct technical terms, proper nouns, numbers, codes, or units.
- Preserve the original language and capitalization intent.
- If a token is uncertain or already valid, leave it unchanged.
- Keep the meaning and word count essentially identical.

Respond with ONLY a JSON object, no markdown and no commentary:
{"corrected_text": "<the corrected text>"}"""


def agent4_correct_ocr(doc_json: dict) -> dict:
    print("\n🟡 Agent 4 — LLM OCR Correction (JSON mode, safe — no DB write)")

    result_json = copy.deepcopy(doc_json)

    # Force Groq into guaranteed-valid JSON output.
    try:
        corrector = llm.bind(response_format={"type": "json_object"})
    except Exception:
        corrector = llm

    fixed_regions = 0

    for page in result_json["pages"]:
        for region in page["layout_regions"]:

            words = (region.get("text") or "").strip()
            if not words:
                region["corrected_text"] = ""
                continue

            corrected = words  # safe default = original OCR text

            # Up to 2 attempts to obtain valid JSON before giving up.
            for _attempt in range(2):
                try:
                    response = corrector.invoke([
                        SystemMessage(content=CORRECTION_SYSTEM_PROMPT),
                        HumanMessage(content=json.dumps({"ocr_text": words}))
                    ])
                    data = _extract_json(response.content)
                    value = data.get("corrected_text")
                    if isinstance(value, str) and value.strip():
                        corrected = value.strip()
                        fixed_regions += 1
                        break
                except Exception:
                    continue  # retry, then fall back to original

            region["corrected_text"] = corrected

    print(f"✅ OCR correction done — {fixed_regions} region(s) corrected (safe mode)")
    return result_json
