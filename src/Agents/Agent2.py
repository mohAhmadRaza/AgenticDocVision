# ════════════════════════════════════════════════════════
# SECTION 7 — AGENT 2: LAYOUT ANALYSIS
# ────────────────────────────────────────────────────────
# Responsibilities:
#   - Load each cleaned page image from Agent 1
#   - Detect layout regions using dilation + contour detection
#   - Find line boundaries using vertical histogram projection
#   - Assign global line numbers across entire page
#   - Return updated doc_json with regions and lines
# ════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════
# HELPERS (Agent 2 Core Utilities)
# ════════════════════════════════════════════════════════

def filter_noise_boxes(boxes, min_area=800):
    """Remove tiny OCR/noise regions"""
    filtered = []
    for (x, y, w, h) in boxes:
        if w * h >= min_area:
            filtered.append((x, y, w, h))
    return filtered


def iou_merge(box1, box2, threshold=0.3):
    """Check if two boxes should be merged (overlap logic)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return False

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou > threshold


def merge_boxes(boxes):
    """Merge overlapping/nearby bounding boxes"""
    merged = []

    for box in boxes:
        x, y, w, h = box
        added = False

        for i in range(len(merged)):
            if iou_merge(merged[i], box):
                mx, my, mw, mh = merged[i]

                nx = min(mx, x)
                ny = min(my, y)
                nw = max(mx + mw, x + w) - nx
                nh = max(my + mh, y + h) - ny

                merged[i] = (nx, ny, nw, nh)
                added = True
                break

        if not added:
            merged.append(box)

    return merged


def sort_reading_order(regions):
    """Top-to-bottom, left-to-right sorting"""
    return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))


def estimate_text_density(binary_crop):
    """Estimate how much text is inside a region"""
    return np.count_nonzero(binary_crop == 255) / binary_crop.size


def _box_iou(a, b):
    """IoU + containment ratio for two [x, y, w, h] boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    inter = (x2 - x1) * (y2 - y1)
    a_area, b_area = aw * ah, bw * bh
    iou      = inter / float(a_area + b_area - inter)
    contain  = inter / float(min(a_area, b_area) + 1e-6)  # smaller box coverage
    return iou, contain


def suppress_overlapping_regions(regions, iou_thresh=0.55, contain_thresh=0.80):
    """
    Remove duplicate/overlapping layout regions so the SAME text is never
    OCR'd twice. Keeps the larger region; drops any later region that
    heavily overlaps (IoU) or is mostly contained inside a kept one.
    """
    ordered = sorted(regions, key=lambda r: r["bbox"][2] * r["bbox"][3],
                     reverse=True)
    kept = []
    for r in ordered:
        duplicate = False
        for k in kept:
            iou, contain = _box_iou(r["bbox"], k["bbox"])
            if iou > iou_thresh or contain > contain_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(r)
    return kept


# ════════════════════════════════════════════════════════
# SECTION 7 — AGENT 2: LAYOUT ANALYSIS
# ────────────────────────────────────────────────────────
# Responsibilities:
#   - Load each cleaned page image from Agent 1
#   - Detect layout regions using dilation + contour detection
#   - Find line boundaries using vertical histogram projection
#   - Assign global line numbers across entire page
#   - Return updated doc_json with regions and lines
# ════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════
# HELPERS (Agent 2 Core Utilities)
# ════════════════════════════════════════════════════════

def filter_noise_boxes(boxes, min_area=800):
    """Remove tiny OCR/noise regions"""
    filtered = []
    for (x, y, w, h) in boxes:
        if w * h >= min_area:
            filtered.append((x, y, w, h))
    return filtered


def iou_merge(box1, box2, threshold=0.3):
    """Check if two boxes should be merged (overlap logic)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return False

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou > threshold


def merge_boxes(boxes):
    """Merge overlapping/nearby bounding boxes"""
    merged = []

    for box in boxes:
        x, y, w, h = box
        added = False

        for i in range(len(merged)):
            if iou_merge(merged[i], box):
                mx, my, mw, mh = merged[i]

                nx = min(mx, x)
                ny = min(my, y)
                nw = max(mx + mw, x + w) - nx
                nh = max(my + mh, y + h) - ny

                merged[i] = (nx, ny, nw, nh)
                added = True
                break

        if not added:
            merged.append(box)

    return merged


def sort_reading_order(regions):
    """Top-to-bottom, left-to-right sorting"""
    return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))


def estimate_text_density(binary_crop):
    """Estimate how much text is inside a region"""
    return np.count_nonzero(binary_crop == 255) / binary_crop.size


def _box_iou(a, b):
    """IoU + containment ratio for two [x, y, w, h] boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    inter = (x2 - x1) * (y2 - y1)
    a_area, b_area = aw * ah, bw * bh
    iou      = inter / float(a_area + b_area - inter)
    contain  = inter / float(min(a_area, b_area) + 1e-6)  # smaller box coverage
    return iou, contain


def suppress_overlapping_regions(regions, iou_thresh=0.55, contain_thresh=0.80):
    """
    Remove duplicate/overlapping layout regions so the SAME text is never
    OCR'd twice. Keeps the larger region; drops any later region that
    heavily overlaps (IoU) or is mostly contained inside a kept one.
    """
    ordered = sorted(regions, key=lambda r: r["bbox"][2] * r["bbox"][3],
                     reverse=True)
    kept = []
    for r in ordered:
        duplicate = False
        for k in kept:
            iou, contain = _box_iou(r["bbox"], k["bbox"])
            if iou > iou_thresh or contain > contain_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(r)
    return kept


# ════════════════════════════════════════════════════════
# AGENT 2 — DOCUMENT LAYOUT ANALYSIS
# ────────────────────────────────────────────────────────
# Primary  : DocLayout-YOLO → semantic regions
#            (title, plain text, table, figure, formula, caption, ...)
# Fallback : OpenCV dilation + contour grouping → text blocks
# Both paths return regions sorted in natural reading order.
# ════════════════════════════════════════════════════════

# DocLayout "abandon" = page furniture (running heads, footers, page
# numbers) — not real content, so we drop it from the index.
_LAYOUT_DROP = {"abandon"}


def _doclayout_regions(img):
    """Detect regions with DocLayout-YOLO. Returns a list of region dicts."""
    det = layout_model.predict(
        img, imgsz=1024, conf=0.2, device=DEVICE, verbose=False
    )[0]

    names = layout_model.names
    regions = []

    for box in det.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id))

        if label in _LAYOUT_DROP:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        if x2 <= x1 or y2 <= y1:
            continue

        regions.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "region_type": label,
            "confidence": round(conf, 3),
            "words_found": 0
        })

    return regions


def _cv_layout_regions(img):
    """OpenCV fallback: group text into blocks via dilation + contours."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = gray.shape[:2]

    # Text → white on black so we can dilate words into blocks.
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 12))
    dilated = cv2.dilate(th, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = filter_noise_boxes(boxes, min_area=800)
    boxes = merge_boxes(boxes)

    regions = []
    for (x, y, w, h) in boxes:
        crop    = th[y:y + h, x:x + w]
        density = estimate_text_density(crop)
        ar      = w / float(h + 1e-5)

        if h > 0.40 * H and density < 0.15:
            rtype = "figure"
        elif ar > 6 and h < 0.05 * H:
            rtype = "title"
        else:
            rtype = "plain text"

        regions.append({
            "bbox": [x, y, w, h],
            "region_type": rtype,
            "confidence": round(float(density), 3),
            "words_found": 0
        })

    return regions


def agent2_layout_analysis(doc_json: dict) -> dict:
    result_json = copy.deepcopy(doc_json)

    print(f"\n🔵 Agent 2 — Layout Analysis (backend: {LAYOUT_BACKEND})")
    print("=" * 60)

    for page in result_json["pages"]:
        # Layout runs on the original page; OCR later uses the clean binary.
        layout_src = page.get("original_path") or page["image_path"]
        img = cv2.imread(layout_src)
        if img is None:
            img = cv2.imread(page["image_path"])
        if img is None:
            page["layout_regions"] = []
            print(f"   ⚠️ Page {page['page_no']}: image unreadable, skipped")
            continue

        regions = []
        if LAYOUT_BACKEND == "doclayout" and layout_model is not None:
            try:
                regions = _doclayout_regions(img)
            except Exception as e:
                print(f"   ⚠️ DocLayout failed on page {page['page_no']} ({e}); using CV")
                regions = []

        # Fallback (or backup if the model found nothing on this page).
        if not regions:
            regions = _cv_layout_regions(img)

        # Drop overlapping/duplicate regions so text isn't OCR'd twice.
        regions = suppress_overlapping_regions(regions)

        # Sort top→bottom, left→right and assign stable reading-order ids.
        regions = sort_reading_order(regions)
        for i, r in enumerate(regions, start=1):
            r["region_id"] = i

        page["layout_regions"] = regions
        print(f"   ✅ Page {page['page_no']}: {len(regions)} regions detected")

    return result_json
