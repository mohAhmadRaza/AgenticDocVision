# ════════════════════════════════════════════════════════
# SECTION 8 — AGENT 3: OCR + INDEXING
# ────────────────────────────────────────────────────────
# Responsibilities:
#   - Loop through every page → region → line from Agent 2
#   - Crop each line from the full page image
#   - Run Tesseract OCR on each line crop
#   - Extract word text, position (bbox), confidence score
#   - Extract character-level bounding boxes
#   - Store each word in ChromaDB with full metadata
#   - Return doc_json with words filled in
# ════════════════════════════════════════════════════════
global bm25_corpus

def uid_safe(name: str) -> str:
    """Sanitize a document name into a filesystem-safe slug."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "doc"


def _compute_orb_descriptor(gray_crop: np.ndarray):
    """
    Compute fixed-length ORB descriptor for a region crop.
    Used by Agent 3 (indexing) and Agent 5 (image query).
    Returns float32 numpy array of 512 values, or None.
    """
    resized  = cv2.resize(gray_crop, (128, 64), interpolation=cv2.INTER_AREA)
    orb      = cv2.ORB_create(nfeatures=256)
    _, descs = orb.detectAndCompute(resized, None)
    if descs is None or len(descs) == 0:
        return None
    flat  = descs.flatten().astype(np.float32)
    fixed = np.zeros(512, dtype=np.float32)
    n     = min(len(flat), 512)
    fixed[:n] = flat[:n]
    return fixed

def _orb_match_score(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """
    Compare two ORB descriptors. Returns similarity score 0.0 to 1.0.
    Higher = more visually similar.
    """
    d1 = desc1[:512].astype(np.uint8).reshape(-1, 32)
    d2 = desc2[:512].astype(np.uint8).reshape(-1, 32)
    if d1.shape[0] == 0 or d2.shape[0] == 0:
        return 0.0
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if not matches:
        return 0.0
    avg_dist = sum(m.distance for m in matches) / len(matches)
    score    = max(0.0, 1.0 - avg_dist / 256.0)
    coverage = min(len(matches) / max(d1.shape[0], d2.shape[0]), 1.0)
    return round(score * 0.7 + coverage * 0.3, 4)

def _template_match_score(query_gray: np.ndarray,
                           region_crop_path: str) -> float:
    """
    Template matching fallback — best for near-exact crops.
    Returns similarity score 0.0 to 1.0.
    """
    region_img = cv2.imread(region_crop_path, cv2.IMREAD_GRAYSCALE)
    if region_img is None:
        return 0.0
    qh, qw = query_gray.shape[:2]
    rh, rw = region_img.shape[:2]
    if qw > rw or qh > rh:
        template = cv2.resize(query_gray, (rw, rh))
        source   = region_img
    else:
        template = query_gray
        source   = region_img
    result   = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return round(float(max_val), 4)


def agent3_ocr_and_index(doc_json: dict) -> dict:
    global word_collection, embedding_collection, visual_index

    result_json = copy.deepcopy(doc_json)
    doc_name = result_json["doc_id"]

    visual_index.clear()

    print("\n🟣 Agent 3 — OCR + Indexing (FIXED SAFE VERSION)")

    for page in result_json["pages"]:
        page_no = page["page_no"]
        img = cv2.imread(page["image_path"])
        if img is None:
            continue

        for region in page["layout_regions"]:
            x, y, w, h = region["bbox"]
            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                region["text"] = ""
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            data = pytesseract.image_to_data(
                gray,
                output_type=pytesseract.Output.DICT,
                config="--oem 3 --psm 6"
            )

            region_words = []

            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])

                if text == "" or conf < 60:
                    continue

                # Word bbox in full-page coordinates → store as
                # left/top/right/bottom (the keys Agent 5 reads).
                wx = x + data["left"][i]
                wy = y + data["top"][i]
                ww = data["width"][i]
                wh = data["height"][i]

                uid = f"{doc_name}__p{page_no}__r{region['region_id']}__w{i}"

                meta = {
                    "document":    doc_name,
                    "book":        doc_name,
                    "page_no":     page_no,
                    "region_id":   region["region_id"],
                    "region_type": region.get("region_type", ""),
                    "line_no":     int(data["line_num"][i]),
                    "word_no":     int(data["word_num"][i]),
                    "text":        text,
                    "confidence":  round(conf, 1),
                    "left":        int(wx),
                    "top":         int(wy),
                    "right":       int(wx + ww),
                    "bottom":      int(wy + wh),
                }

                # STORE WORD (source of truth)
                word_collection.upsert(
                    ids=[uid],
                    documents=[text],
                    metadatas=[meta]
                )

                # STORE EMBEDDING (semantic search)
                embedding_collection.upsert(
                    ids=[uid],
                    documents=[text],
                    embeddings=[embedder.encode(text).tolist()],
                    metadatas=[meta]
                )

                region_words.append(text)

            region["text"] = " ".join(region_words)
            region["words_found"] = len(region_words)

            # ── Build the VISUAL INDEX (powers image search in Agent 5) ──
            desc = _compute_orb_descriptor(gray)
            if desc is not None:
                crop_path = os.path.join(
                    REGION_CROP_DIR, f"{uid_safe(doc_name)}_p{page_no}_r{region['region_id']}.png"
                )
                cv2.imwrite(crop_path, crop)
                visual_index.append({
                    "book":       doc_name,
                    "page_no":    page_no,
                    "region_id":  region["region_id"],
                    "bbox":       [x, y, x + w, y + h],
                    "descriptor": desc,
                    "crop_path":  crop_path,
                })

    print(f"✅ OCR + indexing completed — {word_collection.count()} words, "
          f"{len(visual_index)} visual regions")
    return result_json
