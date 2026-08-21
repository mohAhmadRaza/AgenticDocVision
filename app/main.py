# ╔══════════════════════════════════════════════════════════════════╗
# ║     AgenticDocVision — Beautiful Custom Frontend (Final)        ║
# ║     Word + Line + Semantic + Image Query                        ║
# ╚══════════════════════════════════════════════════════════════════╝

!pip install fastapi uvicorn pyngrok python-multipart -q

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import uvicorn
import shutil, os, threading

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

state = {"ready": False, "filename": "", "word_count": 0, "pages": 0}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = f"/content/uploads/{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        doc_json = run_processing_pipeline(file_path)
        state["ready"]      = True
        state["filename"]   = file.filename
        state["word_count"] = word_collection.count()
        state["pages"]      = doc_json["total_pages"]
        return JSONResponse({
            "success":    True,
            "filename":   file.filename,
            "pages":      doc_json["total_pages"],
            "word_count": word_collection.count()
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/search")
async def search(query: str = Form(...), search_type: str = Form(...)):
    if not state["ready"]:
        return JSONResponse({"success": False, "error": "No document processed yet."})
    if not query.strip():
        return JSONResponse({"success": False, "error": "Empty query."})
    try:
        # semantic uses full phrase; word/line splits into words
        if search_type == "semantic":
            result  = run_query_pipeline(query.strip(), search_type="semantic")
            unique  = dedup_rows(result["results"])
        else:
            stype   = "word" if search_type == "word" else "line"
            words   = query.strip().split()
            all_res = []
            for word in words:
                result = run_query_pipeline(word.strip(), search_type=stype)
                all_res.extend(result["results"])
            unique = dedup_rows(all_res)
            unique = sorted(unique,
                            key=lambda x: (x["book"], x["page_no"],
                                           x["bbox"][1] if x.get("bbox") else 0,
                                           x["bbox"][0] if x.get("bbox") else 0))
        return JSONResponse({
            "success":     True,
            "query":       query,
            "search_type": search_type,
            "total_found": len(unique),
            "results":     unique
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/image-search")
async def image_search(file: UploadFile = File(...)):
    if not state["ready"]:
        return JSONResponse({"success": False,
                             "error": "No document processed yet."})
    try:
        query_path = f"/content/uploads/query_{file.filename}"
        with open(query_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ── STEP 1: try to READ the word(s) in the uploaded image ──────
        extracted = ocr_query_image(query_path)

        if extracted:
            # Image contains text → find that actual WORD in the document.
            rows = []
            for tok in extracted.split():
                tok = tok.strip()
                if len(tok) < 2:
                    continue
                r = agent5_rag_retrieve(tok, search_type="word")
                rows.extend(r["results"])
            rows = dedup_rows(rows)
            rows = sorted(rows, key=lambda x: (
                x["page_no"],
                x["bbox"][1] if x.get("bbox") else 0,
                x["bbox"][0] if x.get("bbox") else 0,
            ))
            return JSONResponse({
                "success":        True,
                "search_type":    "word",       # render as real word hits
                "extracted_text": extracted,
                "total_found":    len(rows),
                "results":        rows
            })

        # ── STEP 2: no readable text (logo/figure) → visual ORB match ──
        results = agent5_rag_retrieve(query_path, search_type="image")
        return JSONResponse({
            "success":        True,
            "search_type":    "image",
            "extracted_text": "",
            "total_found":    results["total_found"],
            "results":        results["results"]
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ══════════════════════════════════════════════════════════════════════
# HTML FRONTEND
# ══════════════════════════════════════════════════════════════════════

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgenticDocVision</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --cream:#FAFAF7; --white:#FFFFFF; --ink:#1A1A1A; --ink2:#444444;
    --ink3:#888888; --accent:#2563EB; --accent2:#DBEAFE; --green:#16A34A;
    --green2:#DCFCE7; --red:#DC2626; --red2:#FEE2E2; --border:#E5E5E5;
    --shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.04);
  }
  html { scroll-behavior:smooth; }
  body { font-family:'DM Sans',sans-serif; background:var(--cream); color:var(--ink); min-height:100vh; line-height:1.6; }

  header { background:var(--white); border-bottom:1px solid var(--border); padding:0 48px; height:64px; display:flex; align-items:center; justify-content:space-between; position:sticky; top:0; z-index:100; }
  .logo { display:flex; align-items:center; gap:10px; }
  .logo-icon { width:32px; height:32px; background:var(--accent); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:16px; }
  .logo-text { font-family:'DM Serif Display',serif; font-size:20px; color:var(--ink); }
  .header-badge { font-size:12px; color:var(--ink3); }

  .hero { padding:64px 48px 48px; max-width:1100px; margin:0 auto; }
  .hero-tag { display:inline-flex; align-items:center; gap:6px; background:var(--accent2); color:var(--accent); font-size:12px; font-weight:600; padding:4px 12px; border-radius:100px; letter-spacing:0.5px; text-transform:uppercase; margin-bottom:20px; }
  .hero h1 { font-family:'DM Serif Display',serif; font-size:clamp(36px,5vw,56px); line-height:1.1; letter-spacing:-1px; margin-bottom:16px; max-width:700px; }
  .hero h1 em { font-style:italic; color:var(--accent); }
  .hero p { font-size:16px; color:var(--ink2); max-width:520px; font-weight:300; margin-bottom:40px; }

  .pipeline { display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:48px; }
  .pipe-step { display:flex; align-items:center; gap:8px; background:var(--white); border:1px solid var(--border); border-radius:8px; padding:8px 14px; font-size:13px; font-weight:500; color:var(--ink2); }
  .pipe-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
  .pipe-arrow { color:var(--ink3); font-size:18px; }

  .main { max-width:1100px; margin:0 auto; padding:0 48px 80px; display:grid; grid-template-columns:360px 1fr; gap:24px; align-items:start; }

  .card { background:var(--white); border:1px solid var(--border); border-radius:16px; box-shadow:var(--shadow); overflow:hidden; }
  .card-header { padding:20px 24px 16px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:10px; }
  .card-icon { width:28px; height:28px; border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:14px; flex-shrink:0; }
  .card-title { font-size:14px; font-weight:600; color:var(--ink); }
  .card-subtitle { font-size:12px; color:var(--ink3); margin-top:1px; }
  .card-body { padding:20px 24px; }

  .upload-zone { border:2px dashed var(--border); border-radius:12px; padding:32px 20px; text-align:center; cursor:pointer; transition:all 0.2s; position:relative; background:var(--cream); }
  .upload-zone:hover,.upload-zone.drag { border-color:var(--accent); background:var(--accent2); }
  .upload-zone input[type="file"] { position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; height:100%; }
  .upload-icon { font-size:32px; margin-bottom:10px; }
  .upload-label { font-size:14px; font-weight:500; color:var(--ink); margin-bottom:4px; }
  .upload-hint { font-size:12px; color:var(--ink3); }
  .upload-filename { margin-top:12px; font-size:13px; color:var(--accent); font-weight:500; display:none; }

  .btn { width:100%; padding:12px; border-radius:10px; border:none; font-family:'DM Sans',sans-serif; font-size:14px; font-weight:600; cursor:pointer; transition:all 0.15s; display:flex; align-items:center; justify-content:center; gap:8px; margin-top:16px; }
  .btn-primary { background:var(--ink); color:white; }
  .btn-primary:hover { background:#333; transform:translateY(-1px); }
  .btn-primary:disabled { background:var(--border); color:var(--ink3); cursor:not-allowed; transform:none; }
  .btn-accent { background:var(--accent); color:white; }
  .btn-accent:hover { background:#1d4ed8; transform:translateY(-1px); }
  .btn-accent:disabled { background:var(--border); color:var(--ink3); cursor:not-allowed; transform:none; }

  .status-box { border-radius:10px; padding:14px 16px; font-size:13px; margin-top:16px; display:none; line-height:1.6; }
  .status-box.success { background:var(--green2); color:var(--green); border:1px solid #bbf7d0; }
  .status-box.error   { background:var(--red2);   color:var(--red);   border:1px solid #fecaca; }
  .status-box.loading { background:var(--accent2); color:var(--accent); border:1px solid #bfdbfe; }
  .status-box .stat-row { display:flex; justify-content:space-between; margin-top:8px; padding-top:8px; border-top:1px solid rgba(0,0,0,0.06); }
  .stat-item { text-align:center; }
  .stat-num { font-size:20px; font-weight:700; font-family:'DM Serif Display',serif; }
  .stat-lbl { font-size:11px; opacity:0.7; }

  .search-wrap { position:relative; margin-bottom:14px; }
  .search-icon { position:absolute; left:14px; top:50%; transform:translateY(-50%); color:var(--ink3); font-size:16px; pointer-events:none; }
  input[type="text"] { width:100%; padding:13px 14px 13px 40px; border:1.5px solid var(--border); border-radius:10px; font-family:'DM Sans',sans-serif; font-size:14px; color:var(--ink); background:var(--cream); outline:none; transition:border-color 0.15s; }
  input[type="text"]:focus { border-color:var(--accent); background:var(--white); }

  .toggle-wrap { display:flex; background:var(--cream); border:1px solid var(--border); border-radius:8px; padding:3px; margin-bottom:14px; }
  .toggle-btn { flex:1; padding:7px 4px; border:none; border-radius:6px; font-family:'DM Sans',sans-serif; font-size:12px; font-weight:500; cursor:pointer; background:transparent; color:var(--ink3); transition:all 0.15s; }
  .toggle-btn.active { background:var(--white); color:var(--ink); box-shadow:0 1px 4px rgba(0,0,0,0.08); }

  .img-query-zone { border:2px dashed var(--border); border-radius:12px; padding:20px; text-align:center; cursor:pointer; transition:all 0.2s; position:relative; background:var(--cream); margin-bottom:14px; display:none; }
  .img-query-zone:hover { border-color:var(--accent); background:var(--accent2); }
  .img-query-zone input[type="file"] { position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; height:100%; }

  .results-header { display:flex; align-items:center; justify-content:space-between; padding:16px 24px; border-bottom:1px solid var(--border); }
  .results-count { font-size:13px; color:var(--ink3); }
  .results-count strong { font-size:22px; font-family:'DM Serif Display',serif; color:var(--ink); margin-right:4px; }
  .results-query { font-size:12px; background:var(--accent2); color:var(--accent); padding:3px 10px; border-radius:100px; font-weight:500; }

  .table-wrap { overflow-x:auto; padding:0 24px 24px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  thead tr { border-bottom:1.5px solid var(--border); }
  th { padding:10px 12px; text-align:left; font-size:11px; font-weight:600; color:var(--ink3); text-transform:uppercase; letter-spacing:0.5px; white-space:nowrap; }
  td { padding:11px 12px; color:var(--ink2); border-bottom:1px solid var(--border); vertical-align:middle; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:var(--cream); }
  .td-num { font-family:'DM Serif Display',serif; color:var(--ink3); font-size:12px; width:32px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:100px; font-size:11px; font-weight:600; }
  .badge-page { background:#FEF3C7; color:#92400E; }
  .badge-line { background:#EDE9FE; color:#5B21B6; }
  .badge-word { background:#DCFCE7; color:#15803D; }
  .badge-score { background:#FEE2E2; color:#DC2626; }
  .word-highlight { font-weight:600; color:var(--ink); background:#FEF08A; padding:1px 6px; border-radius:4px; }
  .conf-bar { display:flex; align-items:center; gap:8px; }
  .conf-track { height:4px; width:48px; background:var(--border); border-radius:2px; overflow:hidden; }
  .conf-fill { height:100%; border-radius:2px; background:var(--green); }

  .empty-state { text-align:center; padding:60px 24px; color:var(--ink3); }
  .empty-icon { font-size:40px; margin-bottom:12px; }
  .empty-title { font-size:16px; font-weight:500; color:var(--ink2); margin-bottom:6px; }
  .empty-hint { font-size:13px; }

  @keyframes spin { to { transform:rotate(360deg); } }
  .spinner { width:16px; height:16px; border:2px solid currentColor; border-top-color:transparent; border-radius:50%; animation:spin 0.7s linear infinite; display:inline-block; }

  .agent-trace { background:var(--cream); border:1px solid var(--border); border-radius:10px; padding:14px 16px; font-size:12px; color:var(--ink2); margin-top:16px; display:none; line-height:1.8; font-family:monospace; }
  .trace-thought { color:#7C3AED; }
  .trace-action  { color:#2563EB; }
  .trace-observe { color:#16A34A; }

  @media (max-width:768px) {
    .main { grid-template-columns:1fr; padding:0 20px 60px; }
    .hero { padding:40px 20px 32px; }
    header { padding:0 20px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🤖</div>
    <span class="logo-text">AgenticDocVision</span>
  </div>
  <span class="header-badge">University of Engineering &amp; Technology, Lahore</span>
</header>

<div class="hero">
  <div class="hero-tag">⚡ 5-Agent AI Pipeline</div>
  <h1>Document <em>Intelligence</em><br>at your fingertips.</h1>
  <p>Upload any research document. Our agentic AI pipeline extracts, indexes, and retrieves every word — with exact page, line, and position data.</p>
  <div class="pipeline">
    <div class="pipe-step"><span class="pipe-dot" style="background:#10B981"></span>Preprocessing</div>
    <span class="pipe-arrow">→</span>
    <div class="pipe-step"><span class="pipe-dot" style="background:#3B82F6"></span>Layout Analysis</div>
    <span class="pipe-arrow">→</span>
    <div class="pipe-step"><span class="pipe-dot" style="background:#8B5CF6"></span>OCR + Index</div>
    <span class="pipe-arrow">→</span>
    <div class="pipe-step"><span class="pipe-dot" style="background:#F59E0B"></span>LLM Correction</div>
    <span class="pipe-arrow">→</span>
    <div class="pipe-step"><span class="pipe-dot" style="background:#EF4444"></span>RAG Retrieval</div>
  </div>
</div>

<div class="main">

  <!-- LEFT PANEL -->
  <div style="display:flex;flex-direction:column;gap:20px;">

    <!-- Upload Card -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon" style="background:#DCFCE7">📄</div>
        <div>
          <div class="card-title">Upload Document</div>
          <div class="card-subtitle">PDF files supported</div>
        </div>
      </div>
      <div class="card-body">
        <div class="upload-zone" id="uploadZone">
          <input type="file" accept=".pdf" id="fileInput" onchange="handleFileSelect(this)">
          <div class="upload-icon">☁️</div>
          <div class="upload-label">Drop your PDF here</div>
          <div class="upload-hint">or click to browse files</div>
          <div class="upload-filename" id="uploadFilename"></div>
        </div>
        <button class="btn btn-primary" id="processBtn" onclick="processDocument()" disabled>
          Process Document
        </button>
        <div class="status-box" id="processStatus"></div>
      </div>
    </div>

    <!-- Search Card -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon" style="background:#DBEAFE">🔍</div>
        <div>
          <div class="card-title">Search</div>
          <div class="card-subtitle">Word · Line · Semantic · Image</div>
        </div>
      </div>
      <div class="card-body">

        <!-- 4 toggle buttons -->
        <div class="toggle-wrap">
          <button class="toggle-btn active" id="toggleWord"     onclick="setMode('word')">📝 Word</button>
          <button class="toggle-btn"        id="toggleLine"     onclick="setMode('line')">📄 Line</button>
          <button class="toggle-btn"        id="toggleSemantic" onclick="setMode('semantic')">🧠 Semantic</button>
          <button class="toggle-btn"        id="toggleImage"    onclick="setMode('image')">🖼️ Image</button>
        </div>

        <!-- Text input (word / line / semantic) -->
        <div class="search-wrap" id="searchWrap">
          <span class="search-icon">🔎</span>
          <input type="text" id="searchInput" placeholder="e.g. neural network, OCR..."
            onkeydown="if(event.key==='Enter') doSearch()">
        </div>

        <!-- Image upload zone (image mode only) -->
        <div class="img-query-zone" id="imageUploadZone">
          <input type="file" accept="image/*" id="queryImageInput"
                 onchange="handleQueryImage(this)">
          <div style="font-size:24px;margin-bottom:6px;">🖼️</div>
          <div class="upload-label" style="font-size:13px;">Upload word screenshot or crop</div>
          <div class="upload-hint">PNG, JPG accepted</div>
          <div class="upload-filename" id="queryImageName"></div>
        </div>

        <button class="btn btn-accent" id="searchBtn" onclick="doSearch()" disabled>
          Search
        </button>
        <div class="agent-trace" id="agentTrace"></div>
      </div>
    </div>

  </div>

  <!-- RIGHT PANEL -->
  <div class="card" id="resultsCard">
    <div class="empty-state" id="emptyState">
      <div class="empty-icon">🗂️</div>
      <div class="empty-title">No search yet</div>
      <div class="empty-hint">Upload a document and search for any word</div>
    </div>
    <div id="resultsContent" style="display:none;">
      <div class="results-header">
        <div class="results-count"><strong id="resultCount">0</strong> results found</div>
        <div class="results-query" id="resultQuery"></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Match</th>
              <th>Document</th>
              <th>Page</th>
              <th>Line</th>
              <th>Word #</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody id="resultsBody"></tbody>
        </table>
      </div>
    </div>
  </div>

</div>

<script>
  let searchMode     = 'word';
  let selectedFile   = null;
  let queryImageFile = null;

  function setMode(mode) {
    searchMode = mode;
    ['Word','Line','Semantic','Image'].forEach(m =>
      document.getElementById('toggle'+m).classList.toggle('active', mode === m.toLowerCase())
    );
    document.getElementById('searchWrap').style.display      = mode === 'image' ? 'none'  : 'block';
    document.getElementById('imageUploadZone').style.display = mode === 'image' ? 'block' : 'none';
  }

  function handleFileSelect(input) {
    selectedFile = input.files[0];
    if (!selectedFile) return;
    document.getElementById('uploadFilename').textContent = '📎 ' + selectedFile.name;
    document.getElementById('uploadFilename').style.display = 'block';
    document.getElementById('processBtn').disabled = false;
  }

  function handleQueryImage(input) {
    queryImageFile = input.files[0];
    if (!queryImageFile) return;
    document.getElementById('queryImageName').textContent = '📎 ' + queryImageFile.name;
    document.getElementById('queryImageName').style.display = 'block';
  }

  // Drag & drop for document
  const zone = document.getElementById('uploadZone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (f && f.name.endsWith('.pdf')) {
      selectedFile = f;
      document.getElementById('uploadFilename').textContent = '📎 ' + f.name;
      document.getElementById('uploadFilename').style.display = 'block';
      document.getElementById('processBtn').disabled = false;
    }
  });

  function showStatus(id, type, html) {
    const el = document.getElementById(id);
    el.className = 'status-box ' + type;
    el.innerHTML = html;
    el.style.display = 'block';
  }

  async function processDocument() {
    if (!selectedFile) return;
    const btn = document.getElementById('processBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Processing...';
    showStatus('processStatus', 'loading', '⏳ Running 4 agents (preprocess → layout → OCR → LLM correction) — this may take 1-2 minutes...');
    const form = new FormData();
    form.append('file', selectedFile);
    try {
      const res  = await fetch('/upload', { method: 'POST', body: form });
      const data = await res.json();
      if (data.success) {
        showStatus('processStatus', 'success',
          `✅ <strong>Document processed!</strong>
          <div class="stat-row">
            <div class="stat-item"><div class="stat-num">${data.pages}</div><div class="stat-lbl">Pages</div></div>
            <div class="stat-item"><div class="stat-num">${data.word_count}</div><div class="stat-lbl">Words Indexed</div></div>
          </div>`);
        document.getElementById('searchBtn').disabled = false;
        btn.innerHTML = '✅ Processed';
      } else {
        showStatus('processStatus', 'error', '❌ ' + data.error);
        btn.disabled = false; btn.innerHTML = 'Process Document';
      }
    } catch(e) {
      showStatus('processStatus', 'error', '❌ ' + e.message);
      btn.disabled = false; btn.innerHTML = 'Process Document';
    }
  }

  async function doSearch() {
    const btn   = document.getElementById('searchBtn');
    const trace = document.getElementById('agentTrace');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Searching...';
    trace.style.display = 'block';

    try {
      // ── IMAGE MODE ────────────────────────────────────
      if (searchMode === 'image') {
        if (!queryImageFile) {
          trace.innerHTML = '<div class="trace-thought">❌ Please upload a query image first.</div>';
          btn.disabled = false; btn.innerHTML = 'Search'; return;
        }
        trace.innerHTML = `
          <div class="trace-line trace-thought">💭 Thought: Image query — reading text from the image (OCR)</div>
          <div class="trace-line trace-action">⚡ Action: OCR the crop, then search the document...</div>`;
        const form = new FormData();
        form.append('file', queryImageFile);
        const res  = await fetch('/image-search', { method: 'POST', body: form });
        const data = await res.json();
        if (data.success) {
          const stype = data.search_type || 'image';
          if (data.extracted_text) {
            trace.innerHTML += `
              <div class="trace-line trace-observe">👁 Observation: Recognized text → "${data.extracted_text}"</div>
              <div class="trace-line trace-observe">✅ Done — ${data.total_found} match(es) found in document</div>`;
          } else {
            trace.innerHTML += `
              <div class="trace-line trace-observe">👁 Observation: No readable text — fell back to ORB visual matching</div>
              <div class="trace-line trace-observe">✅ Done — ${data.total_found} matching region(s) found</div>`;
          }
          const q = data.extracted_text
                    ? `${data.extracted_text} (from image)`
                    : queryImageFile.name;
          renderResults({ query: q, search_type: stype,
                          total_found: data.total_found, results: data.results });
        } else {
          trace.innerHTML += `<div style="color:red">❌ ${data.error}</div>`;
        }

      // ── SEMANTIC MODE ─────────────────────────────────
      } else if (searchMode === 'semantic') {
        const query = document.getElementById('searchInput').value.trim();
        if (!query) { btn.disabled = false; btn.innerHTML = 'Search'; return; }
        trace.innerHTML = `
          <div class="trace-line trace-thought">💭 Thought: Semantic query — computing embedding for "${query}"</div>
          <div class="trace-line trace-action">⚡ Action: Comparing against ${' '}stored embeddings in ChromaDB...</div>`;
        const form = new FormData();
        form.append('query', query);
        form.append('search_type', 'semantic');
        const res  = await fetch('/search', { method: 'POST', body: form });
        const data = await res.json();
        if (data.success) {
          trace.innerHTML += `
            <div class="trace-line trace-observe">👁 Observation: Cosine similarity computed</div>
            <div class="trace-line trace-observe">✅ Done — ${data.total_found} semantically similar result(s)</div>`;
          renderResults(data);
        } else {
          trace.innerHTML += `<div style="color:red">❌ ${data.error}</div>`;
        }

      // ── WORD / LINE MODE ──────────────────────────────
      } else {
        const query = document.getElementById('searchInput').value.trim();
        if (!query) { btn.disabled = false; btn.innerHTML = 'Search'; return; }
        const modeVerb = searchMode === 'line'
          ? 'grouping indexed words into lines'
          : 'exact + fuzzy matching against indexed words';
        trace.innerHTML = `
          <div class="trace-line trace-thought">💭 Thought: ${searchMode === 'line' ? 'Line' : 'Word'} search for "${query}"</div>
          <div class="trace-line trace-action">⚡ Action: ${modeVerb} in ChromaDB...</div>`;
        const form = new FormData();
        form.append('query', query);
        form.append('search_type', searchMode);
        const res  = await fetch('/search', { method: 'POST', body: form });
        const data = await res.json();
        if (data.success) {
          trace.innerHTML += `
            <div class="trace-line trace-observe">👁 Observation: ChromaDB word index searched</div>
            <div class="trace-line trace-observe">✅ Done — ${data.total_found} ${searchMode === 'line' ? 'line(s)' : 'occurrence(s)'} found</div>`;
          renderResults(data);
        } else {
          trace.innerHTML += `<div style="color:red">❌ ${data.error}</div>`;
        }
      }
    } catch(e) {
      trace.innerHTML += `<div style="color:red">❌ ${e.message}</div>`;
    }

    btn.disabled = false; btn.innerHTML = 'Search';
  }

  function renderResults(data) {
    document.getElementById('emptyState').style.display     = 'none';
    document.getElementById('resultsContent').style.display = 'block';
    document.getElementById('resultCount').textContent      = data.total_found;
    document.getElementById('resultQuery').textContent      = '"' + data.query + '"';

    const tbody = document.getElementById('resultsBody');
    if (data.total_found === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:40px;color:#888">
        No results found.</td></tr>`;
      return;
    }

    const dash = '<span style="color:#bbb">—</span>';

    tbody.innerHTML = data.results.map((r, i) => {
      const isImage    = data.search_type === 'image';
      const isLine     = data.search_type === 'line';
      const isSemantic = data.search_type === 'semantic';

      // ── Match column ──────────────────────────────────────────────
      const wordCell   = isImage    ? '🖼️ visual match'
                       : isLine     ? `<span style="color:#444">${r.word}</span>`
                       : isSemantic ? `<span class="word-highlight">${r.word}</span> <span style="font-size:10px;color:#888">(semantic)</span>`
                       :              `<span class="word-highlight">${r.word}</span>`;

      // ── Line / Word# columns (hide values that don't apply) ───────
      const lineCell = (r.line_no && r.line_no > 0)
                       ? `<span class="badge badge-line">l.${r.line_no}</span>` : dash;
      const wordNo   = isImage ? `<span class="badge badge-score">s.${r.score}</span>`
                       : (r.word_no && r.word_no > 0)
                         ? `<span class="badge badge-word">w.${r.word_no}</span>` : dash;

      // ── Score column ──────────────────────────────────────────────
      // Relevance (similarity / visual match) for semantic & image;
      // OCR confidence for exact word / line matches.
      const useScore  = isImage || isSemantic;
      const rawVal    = useScore ? (r.score * 100) : (r.confidence || 0);
      const confVal   = Math.max(0, Math.min(100, rawVal));
      const confLabel = confVal.toFixed(0) + '%';
      const scoreClr  = useScore ? 'var(--accent)' : 'var(--green)';

      return `<tr>
        <td class="td-num">${i+1}</td>
        <td>${wordCell}</td>
        <td style="font-size:12px;color:#666;max-width:120px;overflow:hidden;
            text-overflow:ellipsis;white-space:nowrap" title="${r.book}">${r.book}</td>
        <td><span class="badge badge-page">p.${r.page_no}</span></td>
        <td>${lineCell}</td>
        <td>${wordNo}</td>
        <td>
          <div class="conf-bar">
            <div class="conf-track">
              <div class="conf-fill" style="width:${confVal}%;background:${scoreClr}"></div>
            </div>
            <span style="font-size:11px;color:#666">${confLabel}</span>
          </div>
        </td>
      </tr>`;
    }).join('');
  }
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML


# ══════════════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════════════
def start_server():
    # Direct uvicorn.run ki bajaye config object banayein
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)

    # Isko loop_factory ke bagair run karne ke liye async method call karein
    import asyncio
    asyncio.run(server.serve())

thread = threading.Thread(target=start_server, daemon=True)
thread.start()

import time; time.sleep(2)

ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))
public_url = ngrok.connect(8000)
print()
print("=" * 55)
print("🚀 AgenticDocVision is LIVE!")
print(f"🌐 Open this URL: {public_url}")
print("=" * 55)
