import os
import shutil
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from rag_engine import RAGEngine

# Load environment variables (.env if present)
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes (to support local Vite dev server)
CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "uploads"))
SAMPLE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_data"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize RAGEngine singleton
rag = RAGEngine(db_dir=os.path.join(os.path.dirname(__file__), "chroma_db"))

@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns vector database and model status."""
    try:
        stats = rag.get_stats()
        stats["gemini_api_configured"] = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload_files():
    """Uploads and indexes one or multiple PDF syllabus files."""
    if "files" not in request.files and "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    uploaded_files = request.files.getlist("files") or [request.files.get("file")]
    results = []

    for file_obj in uploaded_files:
        if not file_obj or file_obj.filename == "":
            continue

        filename = file_obj.filename
        if not filename.lower().endswith(".pdf"):
            results.append({
                "filename": filename,
                "success": False,
                "error": "Only PDF files (.pdf) are supported"
            })
            continue

        dest_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            file_obj.save(dest_path)
            index_result = rag.chunk_and_index_pdf(dest_path, filename)
            results.append(index_result)
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            results.append({
                "filename": filename,
                "success": False,
                "error": str(e)
            })

    total_chunks = sum(r.get("chunks_count", 0) for r in results if r.get("success"))
    return jsonify({
        "success": True,
        "results": results,
        "total_new_chunks": total_chunks,
        "stats": rag.get_stats()
    })

@app.route("/api/query", methods=["POST"])
def query_rag():
    """Answers a user question using the RAG pipeline."""
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 4))
    api_key = data.get("api_key", "").strip() or None
    model_name = data.get("model", "gemini-3.5-flash-lite")

    if not query:
        return jsonify({"success": False, "error": "Query cannot be empty"}), 400

    try:
        response = rag.generate_answer(
            query=query,
            top_k=top_k,
            api_key=api_key,
            model_name=model_name
        )
        return jsonify({"success": True, "data": response})
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/documents", methods=["GET"])
def get_documents():
    """Returns list of currently indexed documents."""
    try:
        docs = rag.list_documents()
        return jsonify({"success": True, "documents": docs, "stats": rag.get_stats()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/documents/<filename>", methods=["DELETE"])
def delete_document(filename):
    """Deletes an indexed document from ChromaDB and local disk."""
    try:
        deleted_count = rag.delete_document(filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({
            "success": True,
            "filename": filename,
            "chunks_deleted": deleted_count,
            "stats": rag.get_stats()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/clear", methods=["POST"])
def clear_all():
    """Clears all indexed documents from ChromaDB and upload directory."""
    try:
        rag.clear_all()
        # Clean uploaded files
        for f in os.listdir(UPLOAD_FOLDER):
            fp = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(fp):
                os.remove(fp)
        return jsonify({"success": True, "message": "Knowledge base cleared successfully.", "stats": rag.get_stats()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/chunks", methods=["GET"])
def get_chunks():
    """Returns chunk details for deep inspection."""
    filename = request.args.get("filename")
    try:
        chunks = rag.get_document_chunks(filename=filename)
        return jsonify({"success": True, "chunks": chunks, "count": len(chunks)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/load-sample", methods=["POST"])
def load_sample():
    """Loads and indexes the built-in sample AI/NLP syllabus PDF."""
    sample_pdf_path = os.path.join(SAMPLE_FOLDER, "CS401_AI_NLP_Syllabus.pdf")
    if not os.path.exists(sample_pdf_path):
        from sample_data.generate_sample_pdf import generate_sample_pdf_file
        generate_sample_pdf_file(sample_pdf_path)

    try:
        filename = "CS401_AI_NLP_Syllabus.pdf"
        dest_path = os.path.join(UPLOAD_FOLDER, filename)
        shutil.copyfile(sample_pdf_path, dest_path)
        result = rag.chunk_and_index_pdf(dest_path, filename)
        return jsonify({
            "success": True,
            "result": result,
            "stats": rag.get_stats()
        })
    except Exception as e:
        logger.error(f"Error loading sample: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/study-tools/quiz", methods=["POST"])
def create_quiz():
    """Generates MCQs from the syllabus."""
    data = request.get_json() or {}
    topic = data.get("topic")
    api_key = data.get("api_key", "").strip() or None
    result = rag.generate_quiz(topic=topic, api_key=api_key)
    return jsonify(result)

@app.route("/api/study-tools/summary", methods=["POST"])
def create_summary():
    """Generates an exam revision cheat sheet."""
    data = request.get_json() or {}
    api_key = data.get("api_key", "").strip() or None
    result = rag.generate_study_summary(api_key=api_key)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting SyllabusBot RAG Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
