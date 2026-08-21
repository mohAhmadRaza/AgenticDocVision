# ════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# All required libraries for the entire pipeline
# ════════════════════════════════════════════════════════
import os
import re
from matplotlib import pyplot as plt
from ultralytics import YOLO
import torch
import json
import copy
import shutil
import json as json_module
import numpy as np
import cv2
import pytesseract
import chromadb
import pandas as pd
from PIL import Image
from pathlib import Path
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi

bm25_corpus = []
bm25_index = None

print("✅ Imports done")

# ════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION
# ════════════════════════════════════════════════════════
from google.colab import userdata

GROQ_API_KEY  = userdata.get('GROQ_API_KEY')
UPLOAD_DIR    = "/content/uploads"
OUTPUT_DIR    = "/content/outputs"
VECTOR_DB_DIR = "/content/vector_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"

print("✅ Config defined")

# ════════════════════════════════════════════════════════
# SECTION 3 — MODEL LOADING
# ────────────────────────────────────────────────────────
# Agent 2 uses DocLayout-YOLO — a YOLO model trained specifically on
# document structure (title, plain text, table, figure, formula, ...).
# A generic COCO YOLO (person/car/...) is useless for documents, so we
# load a real layout model and fall back to OpenCV layout detection if
# the weights cannot be downloaded (keeps the pipeline crash-free).
# ════════════════════════════════════════════════════════
DEVICE = 0 if torch.cuda.is_available() else "cpu"

LAYOUT_BACKEND = "cv"   # "doclayout" once the model loads successfully
layout_model   = None
try:
    from doclayout_yolo import YOLOv10
    from huggingface_hub import hf_hub_download

    _layout_weights = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt"
    )
    layout_model   = YOLOv10(_layout_weights)
    LAYOUT_BACKEND = "doclayout"
    print("✅ DocLayout-YOLO loaded (document-structure detection)")
except Exception as _e:
    print(f"⚠️ DocLayout-YOLO unavailable ({_e}).")
    print("   → Falling back to OpenCV contour-based layout detection.")

embedder = SentenceTransformer(EMBEDDING_MODEL)
llm      = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL)
print(f"✅ Models loaded — Layout: {LAYOUT_BACKEND} | LLM: {LLM_MODEL} | Embeddings: {EMBEDDING_MODEL}")

# ════════════════════════════════════════════════════════
# SECTION 4 — CHROMADB INITIALIZATION
# Using EphemeralClient (in-memory) to avoid readonly errors
# ════════════════════════════════════════════════════════
chroma_client   = chromadb.EphemeralClient()
word_collection = chroma_client.get_or_create_collection("document_words")

embedding_collection = chroma_client.get_or_create_collection("document_embeddings")
print(f"✅ ChromaDB ready (in-memory) — Count: {word_collection.count()}")

# ════════════════════════════════════════════════════════
# SECTION 5 — FOLDER SETUP
# ════════════════════════════════════════════════════════
os.makedirs(UPLOAD_DIR,    exist_ok=True)
os.makedirs(OUTPUT_DIR,    exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

REGION_CROP_DIR = "/content/region_crops"
os.makedirs(REGION_CROP_DIR, exist_ok=True)
visual_index = []

print(f"✅ Folders ready")


# ════════════════════════════════════════════════════════
# SECTION 6 — AGENT 1: PREPROCESSING
# ────────────────────────────────────────────────────────
# Responsibilities:
#   - Accept PDF or image file
#   - Convert PDF pages to images (300 DPI)
#   - Apply Otsu binarization (grayscale → black & white)
#   - Remove noise using morphological opening
#   - Detect connected components (individual character blobs)
#   - Save cleaned images to OUTPUT_DIR
#   - Return base doc_json with page metadata
# ════════════════════════════════════════════════════════

def agent1_preprocess(file_path: str) -> dict:
    """
    AGENT 1 — Preprocessing Agent
    ReAct pattern:
      Thought → What file type is this? How many pages?
      Action  → Convert, binarize, denoise, detect components
      Observe → Cleaned image saved, components counted
    """
    file_path       = Path(file_path)
    file_ext        = file_path.suffix.lower()
    doc_id          = file_path.stem
    page_images_raw = []

    # ── THOUGHT: Detect file type and load pages ──────────────────
    if file_ext == ".pdf":
        # Convert each PDF page to high-resolution PIL image (300 DPI)
        pil_pages = convert_from_path(str(file_path), dpi=300)
        for pil_img in pil_pages:
            # Convert PIL (RGB) → NumPy BGR (OpenCV format)
            page_images_raw.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    elif file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        img = cv2.imread(str(file_path))
        if img is None:
            raise ValueError(f"Could not read image: {file_path}")
        page_images_raw.append(img)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Use PDF or image.")

    pages = []
    for page_num, img in enumerate(page_images_raw, start=1):

        # ── ACTION: Convert to grayscale ──────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── ACTION: Otsu binarization → pure black & white ────────
        # Otsu automatically finds the best threshold value
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── ACTION: Noise removal via morphological opening ────────
        # Opening = erosion then dilation
        # Removes small noise dots while preserving text strokes
        kernel = np.ones((2, 2), np.uint8)
        clean  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # ── ACTION: Connected components analysis ──────────────────
        # Groups connected pixels into labeled blobs (characters)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            clean, connectivity=8
        )

        # Filter out tiny components smaller than 50px (noise)
        filtered = np.zeros_like(clean)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 50:
                filtered[labels == i] = 255

        # ── OBSERVE: Save cleaned image (used by OCR — Agent 3) ────
        out_path = os.path.join(OUTPUT_DIR, f"{doc_id}_page_{page_num:03d}_clean.png")
        cv2.imwrite(out_path, filtered)

        # ── Save original page too (used for layout — Agent 2) ─────
        # Layout models work far better on the natural page than on a
        # pure black/white binary. Same dimensions → bboxes stay aligned.
        orig_path = os.path.join(OUTPUT_DIR, f"{doc_id}_page_{page_num:03d}_orig.png")
        cv2.imwrite(orig_path, img)

        h, w = filtered.shape
        pages.append({
            "page_num":       page_num,
            "page_no":        page_num,
            "image_path":     out_path,
            "original_path":  orig_path,
            "width":          w,
            "height":         h,
            "num_components": int(num_labels - 1),
            "layout_regions": []
        })
        print(f"  ✅ Page {page_num}: {num_labels-1} components → {out_path}")

    doc_json = {
        "doc_id":      doc_id,
        "document":    doc_id,
        "total_pages": len(pages),
        "pages":       pages
    }
    print(f"\n📄 Agent 1 done — {len(pages)} page(s) preprocessed for '{doc_id}'")
    return doc_json
