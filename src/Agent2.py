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
