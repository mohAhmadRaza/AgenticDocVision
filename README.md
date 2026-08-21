# 🤖 AgenticDocVision

**A 5-agent AI pipeline for intelligent document understanding, OCR correction, and multi-modal retrieval.**

Upload any scanned PDF or image document → get a fully searchable, structured knowledge base. Search it by **exact word**, **line**, **semantic meaning**, or even by **uploading a screenshot** of a word or logo.

Built as a course project at the **University of Engineering & Technology (UET), Lahore.**

---

## ✨ What it does

Traditional OCR gives you a wall of unstructured, noisy text. AgenticDocVision fixes that by chaining five specialized agents, each responsible for one stage of document intelligence:

| Agent | Responsibility |
|---|---|
| **1 — Preprocessing** | Converts PDF pages to images, applies Otsu binarization, removes noise via morphological opening, filters components via connected-component analysis |
| **2 — Layout Analysis** | Detects semantic regions (title, paragraph, table, figure, formula) using DocLayout-YOLO, with an OpenCV contour-based fallback |
| **3 — OCR + Indexing** | Runs Tesseract OCR on every detected region, extracts word-level bounding boxes and confidence scores, indexes everything into ChromaDB, and builds a visual (ORB-based) index for image search |
| **4 — LLM Correction** | Uses Llama 3.3 70B (via Groq) to repair OCR scanning artifacts — merged/split words, character confusions (`rn`↔`m`, `0`↔`O`) — without rewriting or hallucinating content |
| **5 — RAG Retrieval** | Four retrieval modes: exact/fuzzy word match, line grouping, semantic search via sentence embeddings, and image-based search (OCR-on-crop + ORB visual matching) |

All five agents are orchestrated into a single pipeline and exposed through a **FastAPI backend with a custom web frontend**, deployed publicly via ngrok.

---

## 🏗️ Architecture

```
                 ┌──────────────┐
   PDF / Image → │  Agent 1     │  Preprocessing
                 │  (OpenCV)    │  binarize · denoise · components
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  Agent 2     │  Layout Analysis
                 │ (DocLayout-  │  title / text / table / figure
                 │  YOLO + CV)  │
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  Agent 3     │  OCR + Indexing
                 │ (Tesseract + │  words + bboxes → ChromaDB
                 │  ChromaDB)   │  + ORB visual index
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  Agent 4     │  LLM OCR Correction
                 │ (Llama 3.3   │  fixes scan artifacts,
                 │  via Groq)   │  preserves meaning
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  Agent 5     │  RAG Retrieval
                 │ (Embeddings  │  word · line · semantic · image
                 │  + ORB)      │
                 └──────┬───────┘
                        ▼
              ┌────────────────────┐
              │  FastAPI + Web UI  │  → deployed via ngrok
              └────────────────────┘
```

---

## 🛠️ Tech Stack

- **Computer Vision:** OpenCV, Otsu thresholding, morphological operations, connected components, ORB descriptors
- **Layout Detection:** DocLayout-YOLO (YOLOv10 fine-tuned on document structure), Ultralytics
- **OCR:** Tesseract (`pytesseract`)
- **Vector Search / Storage:** ChromaDB (in-memory/ephemeral)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **LLM Correction:** Llama 3.3 70B via Groq + LangChain
- **Backend:** FastAPI, Uvicorn
- **Deployment:** ngrok (public tunnel), designed to run on Google Colab
- **Frontend:** Vanilla HTML/CSS/JS (custom-built, no framework)

---

## 📸 Demo

> _Add a screenshot or short GIF of the web UI here — drag-and-drop upload, search results table, and the agent trace panel all make for a great visual._

```
docs/screenshots/upload.png
docs/screenshots/search-results.png
```

---

## 🚀 Getting Started

### Prerequisites

This project was built and tested on **Google Colab** (for free GPU/CPU access and easy system-package installs). It can also run locally on Linux/WSL with the same system dependencies.

**System packages** (not pip-installable):
```bash
apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils
```

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/AgenticDocVision.git
cd AgenticDocVision
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your environment variables

```bash
cp .env.example .env
```

Then open `.env` and fill in:

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) — free tier available |
| `NGROK_AUTH_TOKEN` | [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken) — free tier available |

> ⚠️ **Never commit your real `.env` file.** It's already excluded via `.gitignore` — only `.env.example` (with placeholder values) should ever be pushed.

### 4. Run the pipeline

**Option A — Notebook (Colab):**
Open `notebooks/AgenticDocVision.ipynb` in Google Colab and run all cells top to bottom.

**Option B — As a script/app:**
```bash
python app/main.py
```
This starts the FastAPI server, exposes it via ngrok, and prints a public URL you can open in any browser.

### 5. Use it

1. Open the printed ngrok URL
2. Drag and drop a PDF into the upload zone → click **Process Document**
3. Switch between **Word / Line / Semantic / Image** search modes
4. Search a term, or upload a screenshot of a word/logo to find it inside the document

---

## 🔍 Search Modes Explained

| Mode | How it works |
|---|---|
| **Word** | Exact + fuzzy string matching (`SequenceMatcher`) against every indexed word |
| **Line** | Groups words by their detected line/region and matches full lines containing the query |
| **Semantic** | Encodes the query with `all-MiniLM-L6-v2` and finds the closest matches by cosine similarity in ChromaDB |
| **Image** | First tries OCR on the uploaded crop to recognize the word directly; if no text is found (e.g. a logo), falls back to ORB feature matching against a visual index of every region in the document |

---

## 📂 Project Structure

```
AgenticDocVision/
├── src/
│   ├── agents/
│   │   ├── agent1_preprocessing.py
│   │   ├── agent2_layout.py
│   │   ├── agent3_ocr_index.py
│   │   ├── agent4_llm_correction.py
│   │   └── agent5_rag_retrieval.py
│   ├── orchestrator.py         # chains all 5 agents together
│   ├── config.py                # paths, model names, ChromaDB init
│   └── utils.py
├── app/
│   ├── main.py                  # FastAPI routes + ngrok launch
│   └── frontend.py               # HTML/CSS/JS web UI
├── notebooks/
│   └── AgenticDocVision.ipynb
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🎯 Why this project

Scanned documents — research papers, forms, archival records — are notoriously hard to search: plain OCR output is noisy, unstructured, and gives no sense of position or content type. AgenticDocVision addresses this by combining classical computer vision preprocessing, layout-aware detection, OCR, LLM-based error correction, and multi-modal retrieval into one pipeline — turning a static scanned page into a structured, queryable, RAG-ready knowledge source. This has direct real-world relevance to archive digitization, academic search, and document-heavy enterprise workflows.

---

## 📌 Known Limitations

- ChromaDB is currently **ephemeral (in-memory)** — the index resets between sessions/documents. Swapping in a persistent ChromaDB client would allow multi-document, multi-session search.
- DocLayout-YOLO requires downloading pretrained weights on first run; if unavailable, the pipeline automatically falls back to an OpenCV-based layout detector.
- OCR accuracy depends heavily on scan quality; the LLM correction agent (Agent 4) mitigates but doesn't eliminate this.

---

## 👤 Author

**Ahmad raza** — Software Engineering student, University of Engineering & Technology (UET), Lahore

---

## 📄 License

This project was built for academic purposes as part of a university coursework project. Feel free to fork and adapt for learning.
