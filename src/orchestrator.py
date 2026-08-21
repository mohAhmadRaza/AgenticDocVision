# ════════════════════════════════════════════════════════
# SECTION 11 — ORCHESTRATOR
# ────────────────────────────────────────────────────────
# Chains all 5 agents into two easy-to-call functions:
#   run_processing_pipeline(file_path) → processes document
#   run_query_pipeline(query, type)    → searches document
# ════════════════════════════════════════════════════════

def run_processing_pipeline(file_path: str) -> dict:

    global chroma_client, word_collection, embedding_collection, visual_index

    print("🔄 Resetting system...")

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fresh in-memory DB + visual index for each document.
    chroma_client        = chromadb.EphemeralClient()
    word_collection      = chroma_client.get_or_create_collection("document_words")
    embedding_collection = chroma_client.get_or_create_collection("document_embeddings")
    visual_index = []

    print("\n🚀 START PIPELINE")

    # 1
    doc_json = agent1_preprocess(file_path)

    # 2
    doc_json = agent2_layout_analysis(doc_json)

    # 3
    doc_json = agent3_ocr_and_index(doc_json)

    # 4 (SAFE LLM correction ONLY)
    doc_json = agent4_correct_ocr(doc_json)

    print("\n✅ COMPLETE PIPELINE FINISHED")
    print("📊 Words indexed:", word_collection.count())

    return doc_json

print()
print("✅ AgenticDocVision — All agents and orchestrator loaded!")
print()
print("   📌 USAGE:")
print("   ─────────────────────────────────────────────────────────")
print("   Step 1 → Upload PDF to /content/uploads/")
print("   Step 2 → run_processing_pipeline('/content/uploads/file.pdf')")
print("   Step 3 → run_query_pipeline('word', search_type='word')")
print("            run_query_pipeline('word', search_type='line')")
