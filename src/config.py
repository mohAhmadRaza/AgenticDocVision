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
