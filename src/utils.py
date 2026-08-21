# ╔══════════════════════════════════════════════════════════════════╗
# ║          AgenticDocVision — Installation Cell                   ║
# ║          University of Engineering & Technology, Lahore         ║
# ║  Run this ONCE per Colab session before anything else           ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── System dependencies (Tesseract OCR engine + PDF converter) ────────
!apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils -q
!pip install -q rank-bm25 layoutparser opencv-python scikit-learn
# ── Python libraries ──────────────────────────────────────────────────
!pip install -q \
    "numpy==1.26.4" \
    "chromadb==0.4.24" \
    pytesseract \
    opencv-python-headless \
    pdf2image \
    Pillow \
    sentence-transformers \
    langchain \
    langchain-groq \
    langchain-community \
    groq

# ── Force correct numpy version ───────────────────────────────────────
!pip install --no-cache-dir numpy==1.26.4 -q
!pip install -q ultralytics
# ── Document-layout YOLO (title/text/table/figure/formula detection) ──
!pip install -q doclayout-yolo huggingface_hub

print("✅ Installation complete!")
