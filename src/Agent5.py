# ════════════════════════════════════════════════════════
# SECTION 10 — AGENT 5: RAG RETRIEVAL (UPDATED)
# ────────────────────────────────────────────────────────
# search_type = "word"     → exact/fuzzy word match (unchanged)
# search_type = "line"     → line grouping (unchanged)
# search_type = "semantic" → embedding similarity search (NEW)
# ════════════════════════════════════════════════════════

from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _pack(query, results, search_type):
    """Uniform response envelope for every search mode."""
    return {
        "query":       query,
        "results":     results,
        "type":        search_type,
        "search_type": search_type,
        "total_found": len(results),
    }


def _meta_bbox(meta):
    return [meta.get("left", 0), meta.get("top", 0),
            meta.get("right", 0), meta.get("bottom", 0)]


def dedup_rows(rows, x_tol=25, y_tol=15):
    """
    Collapse duplicate hits caused by overlapping layout regions.
    Two rows are 'the same hit' when they are the same word, on the same
    page, at essentially the same pixel location (region_id ignored).
    Proximity-based (not grid-bucketed) so small pixel jitter between
    overlapping regions can't split a duplicate across bucket edges.
    """
    kept = []
    for r in rows:
        bx = r.get("bbox") or [0, 0, 0, 0]
        w  = str(r.get("word", "")).lower().strip()
        is_dup = False
        for k in kept:
            kb = k.get("bbox") or [0, 0, 0, 0]
            if (k.get("book", "")  == r.get("book", "")
                and k.get("page_no", 0) == r.get("page_no", 0)
                and str(k.get("word", "")).lower().strip() == w
                and abs(kb[0] - bx[0]) <= x_tol
                and abs(kb[1] - bx[1]) <= y_tol):
                is_dup = True
                break
        if not is_dup:
            kept.append(r)
    return kept


def ocr_query_image(path: str) -> str:
    """
    Read the text out of an uploaded query image (a screenshot/crop of a
    word or phrase). This is what makes image search find the actual WORD
    instead of doing fuzzy visual feature matching. Returns "" if no text.
    """
    img = cv2.imread(path)
    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale small crops — Tesseract is far more accurate on larger text.
    h, w = gray.shape[:2]
    if max(h, w) < 600:
        scale = 600.0 / max(h, w)
        gray  = cv2.resize(gray, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # psm 7 = single text line; fall back to psm 6 (block) if empty.
    for psm in (7, 6):
        text = pytesseract.image_to_string(
            th, config=f"--oem 3 --psm {psm}"
        ).strip()
        if text:
            return " ".join(text.split())  # normalise whitespace/newlines
    return ""


def _row_from_meta(doc, meta, score):
    """Build a frontend-ready result row from stored word metadata."""
    return {
        "word":       doc,
        "score":      round(float(score), 3),
        "book":       meta.get("book", meta.get("document", "")),
        "page_no":    meta.get("page_no", 0),
        "line_no":    meta.get("line_no", 0),
        "word_no":    meta.get("word_no", 0),
        "region_id":  meta.get("region_id", 0),
        "confidence": meta.get("confidence", 0),
        "bbox":       _meta_bbox(meta),
    }


def agent5_rag_retrieve(query, search_type="word"):
    global word_collection, embedding_collection, visual_index

    print(f"\n🔎 Search: {search_type} | Query: {query}")

    # =========================
    # WORD SEARCH (exact + close fuzzy match)
    # =========================
    if search_type == "word":
        data = word_collection.get(include=["documents", "metadatas"])
        results = []
        q = query.lower().strip()

        for doc, meta in zip(data["documents"], data["metadatas"]):
            sim = similarity(doc, q)
            if doc.lower() == q or sim >= 0.85:
                score = 1.0 if doc.lower() == q else sim
                results.append(_row_from_meta(doc, meta, score))

        results.sort(key=lambda r: -r["score"])
        return _pack(query, results, "word")

    # =========================
    # LINE SEARCH (group words by region/line, return matching lines)
    # =========================
    if search_type == "line":
        data = word_collection.get(include=["documents", "metadatas"])

        grouped = {}   # key → {"words": [...], "meta": first_meta}
        for doc, meta in zip(data["documents"], data["metadatas"]):
            key = (meta.get("book", meta.get("document", "")),
                   meta.get("page_no", 0),
                   meta.get("region_id", 0),
                   meta.get("line_no", 0))
            entry = grouped.setdefault(key, {"words": [], "meta": meta})
            entry["words"].append(doc)

        q = query.lower().strip()
        results = []
        for entry in grouped.values():
            line_text = " ".join(entry["words"])
            if q and q not in line_text.lower():
                continue
            meta = entry["meta"]
            results.append({
                "word":       line_text,          # full line in the "match" column
                "text":       line_text,
                "score":      1.0,
                "book":       meta.get("book", meta.get("document", "")),
                "page_no":    meta.get("page_no", 0),
                "line_no":    meta.get("line_no", 0),
                "word_no":    0,
                "region_id":  meta.get("region_id", 0),
                "confidence": meta.get("confidence", 0),
                "bbox":       _meta_bbox(meta),
            })

        results.sort(key=lambda r: (r["page_no"], r["region_id"], r["line_no"]))
        return _pack(query, results, "line")

    # =========================
    # SEMANTIC SEARCH (embedding similarity)
    # =========================
    if search_type == "semantic":
        if embedding_collection.count() == 0:
            return _pack(query, [], "semantic")

        q_emb = embedder.encode(query).tolist()
        res = embedding_collection.query(
            query_embeddings=[q_emb],
            n_results=min(10, embedding_collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        results = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            score = max(0.0, 1 - dist / 2)   # cosine distance → similarity
            results.append(_row_from_meta(doc, meta, score))

        return _pack(query, results, "semantic")

    # =========================
    # IMAGE SEARCH (ORB descriptor match against the visual index)
    # =========================
    if search_type == "image":
        q_img = cv2.imread(query, cv2.IMREAD_GRAYSCALE)
        if q_img is None:
            return _pack(query, [], "image")
        q_img  = cv2.resize(q_img, (128, 64))
        q_desc = _compute_orb_descriptor(q_img)
        if q_desc is None:
            return _pack(query, [], "image")

        results = []
        for entry in visual_index:
            score = _orb_match_score(q_desc, entry["descriptor"])
            if score > 0.3:
                results.append({
                    "word":       "",
                    "score":      round(float(score), 3),
                    "book":       entry.get("book", ""),
                    "page_no":    entry["page_no"],
                    "line_no":    0,
                    "word_no":    0,
                    "region_id":  entry.get("region_id", 0),
                    "confidence": round(float(score) * 100, 1),
                    "bbox":       entry["bbox"],
                })

        results.sort(key=lambda x: -x["score"])
        return _pack(query, results[:20], "image")

    raise ValueError("Invalid search_type")


# ════════════════════════════════════════════════════════
# DISPLAY — updated to show semantic score label
# ════════════════════════════════════════════════════════
def show_word_crop(result):
    """
    Displays the actual word crop using stored bbox.
    """

    page_path = os.path.join(
        OUTPUT_DIR,
        f"{result['book']}_page_{result['page_no']:03d}_clean.png"
    )

    img = cv2.imread(page_path)

    if img is None:
        print("⚠️ Page image not found.")
        return

    left, top, right, bottom = result["bbox"]

    pad = 8

    x1 = max(0, left - pad)
    y1 = max(0, top - pad)
    x2 = min(img.shape[1], right + pad)
    y2 = min(img.shape[0], bottom + pad)

    crop = img[y1:y2, x1:x2]

    plt.figure(figsize=(4,2))
    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def display_results(results):
    print("\n" + "="*70)

    if results["search_type"] == "semantic":
        print(f"🧠 SEMANTIC SEARCH RESULTS — '{results['query']}'")
    else:
        print(f"📋 SEARCH RESULTS — '{results['query']}'")

    print("="*70)

    if results["total_found"] == 0:
        print("❌ No results found")
        return

    for i, r in enumerate(results["results"][:20], 1):
        if "text" in r:
            # line mode
            print(f"{i}. [{r['book']}] P{r['page_no']} → {r['text']}")
        else:
            # word or semantic mode
            label = "semantic score" if results["search_type"] == "semantic" else "score"
            print(f"{i}. [{r['book']}] P{r['page_no']} → '{r['word']}' ({label}: {r['score']})")
            show_word_crop(r)


# ════════════════════════════════════════════════════════
# ORCHESTRATOR — updated run_query_pipeline
# ════════════════════════════════════════════════════════
def run_query_pipeline(query, search_type="word"):
    results = agent5_rag_retrieve(query, search_type)

    print("\n================ RESULTS ================")
    for r in results["results"][:10]:
        print(r)

    return results


print()
print("✅ Agent 5 updated — 3 search types available!")
print()
print("   📌 USAGE:")
print("   ─────────────────────────────────────────────────────────")
print("   run_query_pipeline('student',   search_type='word')")
print("   run_query_pipeline('student',   search_type='line')")
print("   run_query_pipeline('student',   search_type='semantic')")
print("   ─────────────────────────────────────────────────────────")
