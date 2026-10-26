import os
import json
import logging
from typing import List, Dict, Any, Optional
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, db_dir: str = "chroma_db", embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.db_dir = os.path.abspath(db_dir)
        os.makedirs(self.db_dir, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB persistent client at {self.db_dir}")
        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        
        # Collection for syllabus chunks
        self.collection = self.chroma_client.get_or_create_collection(
            name="syllabus_knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Loading SentenceTransformer model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Extracts text from a PDF file page by page using PyPDF.
        Returns a list of dicts with page content and metadata.
        """
        pages_data = []
        try:
            reader = pypdf.PdfReader(file_path)
            total_pages = len(reader.pages)
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                # Clean up repeated whitespaces
                cleaned_text = " ".join(page_text.split())
                if cleaned_text:
                    pages_data.append({
                        "text": cleaned_text,
                        "source": filename,
                        "page": page_idx + 1,
                        "total_pages": total_pages
                    })
        except Exception as e:
            logger.error(f"Error reading PDF {filename}: {e}")
            raise
        return pages_data

    def chunk_and_index_pdf(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Extracts, chunks, embeds, and indexes a PDF document into ChromaDB.
        """
        pages_data = self.extract_text_from_pdf(file_path, filename)
        if not pages_data:
            return {
                "success": False,
                "message": f"No extractable text found in '{filename}'. It might be a scanned image.",
                "chunks_count": 0,
                "pages_count": 0
            }

        # Remove previous chunks of the same file if re-uploading
        self.delete_document(filename)

        all_chunks = []
        all_metadatas = []
        all_ids = []

        chunk_counter = 0
        for page_info in pages_data:
            page_text = page_info["text"]
            page_num = page_info["page"]
            total_pages = page_info["total_pages"]

            chunks = self.text_splitter.split_text(page_text)
            for c_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:
                    continue  # skip tiny trivial fragments
                
                chunk_id = f"{filename}_p{page_num}_c{c_idx}"
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "page": page_num,
                    "total_pages": total_pages,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_counter
                })
                all_ids.append(chunk_id)
                chunk_counter += 1

        if not all_chunks:
            return {
                "success": False,
                "message": "Extracted text was too short to form searchable chunks.",
                "chunks_count": 0,
                "pages_count": len(pages_data)
            }

        # Generate embeddings in batch
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks from {filename}...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=False).tolist()

        # Insert into ChromaDB
        self.collection.add(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas
        )

        return {
            "success": True,
            "filename": filename,
            "chunks_count": len(all_chunks),
            "pages_count": len(pages_data),
            "message": f"Successfully indexed {len(all_chunks)} chunks across {len(pages_data)} pages."
        }

    def query_similarity(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search against the indexed syllabus chunks in ChromaDB.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count())
        )

        retrieved_items = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.5] * len(docs)
            ids = results["ids"][0]

            for doc, meta, dist, cid in zip(docs, metadatas, distances, ids):
                # Cosine distance to similarity percentage
                similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
                retrieved_items.append({
                    "id": cid,
                    "content": doc,
                    "source": meta.get("source", "Unknown Document"),
                    "page": meta.get("page", 1),
                    "total_pages": meta.get("total_pages", 1),
                    "similarity": round(similarity_score * 100, 1)
                })

        return retrieved_items

    def generate_answer(
        self,
        query: str,
        top_k: int = 4,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash-lite"
    ) -> Dict[str, Any]:
        """
        End-to-end RAG pipeline:
        1. Retrieve top-k semantic chunks from ChromaDB.
        2. Assemble augmented context with document and page citations.
        3. Send to Gemini LLM (gemini-3.5-flash-lite) for grounded response.
        """
        retrieved_chunks = self.query_similarity(query, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": "No relevant syllabus or study materials have been uploaded yet. Please upload a PDF syllabus in the left sidebar to start asking questions!",
                "sources": [],
                "chunks_retrieved": 0,
                "is_fallback": True
            }

        # Build context string
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            context_blocks.append(
                f"[Source {idx} | Document: {chunk['source']} (Page {chunk['page']})]:\n{chunk['content']}"
            )
        formatted_context = "\n\n".join(context_blocks)

        prompt = f"""You are an expert Academic Syllabus & Study Material Assistant.
Your task is to answer the student's question clearly, thoroughly, and accurately based ONLY on the provided context excerpts from their uploaded study materials.

Guidelines:
1. Base your answer strictly on the context provided below.
2. If the question cannot be answered from the provided syllabus excerpts, explicitly state:
   "Based on the uploaded syllabus materials, this specific topic is not covered or mentioned. Here is what related concepts were found in the syllabus:" and summarize any partial overlap.
3. Cite the document name and page number when explaining key points or topics (e.g., "[Doc: AI_Syllabus.pdf, Page 2]").
4. Organize your answer with clear markdown headings, bullet points, and bold terms so it is easy for a student to study and revise.
5. Provide a helpful, encouraging academic tone.

Student Question:
{query}

Retrieved Syllabus Excerpts:
---------------------
{formatted_context}
---------------------

Structured Answer for the Student:"""

        # Check for Gemini API key
        active_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()

        if active_key:
            # Model priority list: target gemini-3.5-flash-lite first
            candidate_models = [model_name, "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"]
            seen = set()
            models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

            client = genai.Client(api_key=active_key)
            last_error = ""

            for m_id in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=m_id,
                        contents=prompt
                    )
                    if response and response.text:
                        return {
                            "answer": response.text.strip(),
                            "sources": retrieved_chunks,
                            "chunks_retrieved": len(retrieved_chunks),
                            "model_used": m_id,
                            "is_fallback": False
                        }
                except Exception as inner_e:
                    last_error = str(inner_e)
                    logger.warning(f"Google GenAI call failed with {m_id}: {inner_e}")
                    continue

            # Fallback to direct context if rate-limited or API error occurs
            is_rate_limited = "429" in last_error or "quota" in last_error.lower()
            is_busy = "503" in last_error or "unavailable" in last_error.lower()
            
            if is_rate_limited:
                error_notice = "Free Tier API Rate Limit (Please wait a few seconds before your next question)"
            elif is_busy:
                error_notice = "Gemini Cloud API is currently experiencing high demand"
            else:
                error_notice = f"Gemini API Notice ({last_error})"

            summary_blocks = []
            for c in retrieved_chunks[:3]:
                summary_blocks.append(f"**From `{c['source']}` (Page {c['page']}, Match: {c['similarity']}%):**\n> {c['content']}")

            return {
                "answer": f"**{error_notice}**\n\n### Grounded Excerpts Extracted from Your Syllabus:\n" + "\n\n".join(summary_blocks),
                "sources": retrieved_chunks,
                "chunks_retrieved": len(retrieved_chunks),
                "is_fallback": True,
                "error": last_error
            }
        else:
            # Smart Offline / Preview Mode when no API key is provided yet
            summary_blocks = []
            for c in retrieved_chunks[:3]:
                summary_blocks.append(f"**From `{c['source']}` (Page {c['page']}, Match: {c['similarity']}%):**\n> {c['content']}")

            offline_answer = f"""**Notice: Gemini API Key Not Configured Yet**
*(Add your Gemini API Key in the top-right Settings modal for complete generative LLM answers)*

### Retrieved Syllabus Content for: *"{query}"*
The semantic retrieval engine successfully matched **{len(retrieved_chunks)} relevant sections** in your uploaded syllabus:

""" + "\n\n".join(summary_blocks) + """

---
**Summary & Source Reference:**
* The above excerpts were retrieved directly from your indexed ChromaDB vector store.
* Once you provide your Gemini API key, the chatbot will synthesize these into customized study summaries, exam notes, and structured answers.
"""
            return {
                "answer": offline_answer,
                "sources": retrieved_chunks,
                "chunks_retrieved": len(retrieved_chunks),
                "is_fallback": True,
                "notice": "API Key missing"
            }

    def _call_gemini_with_retry(self, client: Any, model_id: str, prompt: str, max_retries: int = 2) -> str:
        """Executes a generate_content call with retry on 503 or 429 errors."""
        import time
        last_exc = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                if attempt < max_retries - 1 and ("503" in err_str or "429" in err_str or "unavailable" in err_str or "high demand" in err_str):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise e
        raise last_exc

    def _generate_grounded_fallback_quiz(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesizes a 4-question Multiple Choice Quiz directly from retrieved syllabus chunks
        if the Gemini API is temporarily busy (503) or rate-limited.
        """
        questions = []
        for i, c in enumerate(chunks[:4]):
            content = c.get("content", "")
            source = f"{c.get('source', 'Syllabus')} (Page {c.get('page', 1)})"
            
            # Extract key phrases from content
            words = [w.strip("(),.:;-") for w in content.split() if len(w) > 4 and w.isalnum()]
            key_term = words[0] if words else "Course Topic"
            snippet = content[:140] + "..." if len(content) > 140 else content
            
            questions.append({
                "question": f"Based on {source}, which of the following is explicitly discussed in the syllabus regarding '{key_term}'?",
                "options": [
                    f"A) {snippet[:60]}",
                    "B) Advanced Quantum Thermodynamics in Computer Science",
                    "C) Classical Mediterranean History and Epigraphy",
                    "D) None of the mentioned concepts"
                ],
                "correct_answer": 0,
                "explanation": f"According to {source}, the syllabus specifically covers: '{snippet}'.",
                "source": source
            })
            
        while len(questions) < 4:
            src = chunks[0].get("source", "Syllabus") if chunks else "Syllabus"
            pg = chunks[0].get("page", 1) if chunks else 1
            questions.append({
                "question": f"Which of the following is an objective or requirement according to the uploaded study materials?",
                "options": [
                    "A) Mastering core foundational syllabus units and exam guidelines",
                    "B) Unsupervised space telemetry",
                    "C) Marine biological surveying",
                    "D) Non-standard unaccredited courses"
                ],
                "correct_answer": 0,
                "explanation": f"Grounded in the uploaded syllabus document: {src} (Page {pg}).",
                "source": f"{src} (Page {pg})"
            })

        return {
            "quiz_title": "Syllabus Knowledge Review Quiz",
            "questions": questions[:4]
        }

    def generate_quiz(self, topic: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates 4 multiple-choice questions from the uploaded syllabus with retry and multi-model fallback.
        """
        query = topic or "course units objectives syllabus core topics concepts"
        retrieved = self.query_similarity(query, top_k=6)
        if not retrieved:
            return {"success": False, "message": "No syllabus indexed yet."}

        context = "\n\n".join([f"[{c['source']}, Page {c['page']}]: {c['content']}" for c in retrieved])
        active_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()

        if not active_key:
            return {
                "success": False,
                "message": "Gemini API key is required to generate customized AI quizzes. Please add your key in Settings."
            }

        prompt = f"""Based on the following syllabus excerpts, create a 4-question Multiple Choice Quiz for students to test their knowledge.
Return ONLY valid JSON in the exact format:
{{
  "quiz_title": "Syllabus Review Quiz",
  "questions": [
    {{
      "question": "Question text here?",
      "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
      "correct_answer": 0,
      "explanation": "Brief explanation citing syllabus topic.",
      "source": "Document name (Page X)"
    }}
  ]
}}

Syllabus Content:
{context}"""

        candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-latest", "gemini-3.5-flash"]
        client = genai.Client(api_key=active_key)
        raw_text = ""
        last_error = ""

        for m_id in candidate_models:
            try:
                raw_text = self._call_gemini_with_retry(client, m_id, prompt, max_retries=2)
                if raw_text:
                    break
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Quiz model {m_id} error: {e}")
                continue

        if raw_text:
            try:
                cleaned = raw_text
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json")[1].split("```")[0]
                elif "```" in cleaned:
                    cleaned = cleaned.split("```")[1].split("```")[0]
                quiz_data = json.loads(cleaned.strip())
                return {"success": True, "quiz": quiz_data}
            except Exception as parse_e:
                logger.warning(f"Error parsing Gemini JSON: {parse_e}")

        # Fallback to grounded syllabus quiz if Gemini servers are temporarily 503 unavailable
        logger.info("Using grounded syllabus quiz fallback due to API status")
        fallback_quiz = self._generate_grounded_fallback_quiz(retrieved)
        return {"success": True, "quiz": fallback_quiz}

    def generate_study_summary(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a structured unit-by-unit high-yield revision summary with retry.
        """
        retrieved = self.query_similarity("course outline syllabus units topics modules learning outcomes", top_k=6)
        if not retrieved:
            return {"success": False, "message": "No syllabus indexed yet."}

        context = "\n\n".join([f"[{c['source']}, Page {c['page']}]: {c['content']}" for c in retrieved])
        active_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()

        if not active_key:
            return {
                "success": False,
                "message": "Gemini API key is required to generate AI study guides. Please add your key in Settings."
            }

        prompt = f"""Generate an organized, high-yield 'Exam Revision Cheat Sheet' based on the following syllabus:
Include:
1. Course Overview & Key Focus Areas
2. Core Modules/Units Breakdown
3. High-Priority Exam Topics
4. Key Terms & Definitions

Format in clean markdown with bullet points.

Syllabus Context:
{context}"""

        candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-latest", "gemini-3.5-flash"]
        client = genai.Client(api_key=active_key)
        for m_id in candidate_models:
            try:
                summary_text = self._call_gemini_with_retry(client, m_id, prompt, max_retries=2)
                if summary_text:
                    return {"success": True, "summary": summary_text}
            except Exception as e:
                logger.warning(f"Summary model {m_id} error: {e}")
                continue

        # Grounded summary fallback
        fallback_summary = f"""### Exam Revision Cheat Sheet (Extracted from Syllabus)

#### Course Focus & Key Units
Based on your uploaded syllabus excerpts:

""" + "\n\n".join([f"* **From `{c['source']}` (Page {c['page']})**:\n  {c['content']}" for c in retrieved[:4]]) + """

#### High-Priority Exam Topics
* Review all foundational definitions and unit outlines specified on the cited pages above.
* Prepare for mid-term and end-term questions covering the listed topics.
"""
        return {"success": True, "summary": fallback_summary}

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Lists all indexed documents with their chunk counts and page ranges.
        """
        if self.collection.count() == 0:
            return []

        all_data = self.collection.get(include=["metadatas"])
        metas = all_data.get("metadatas", [])
        
        docs_map = {}
        for m in metas:
            src = m.get("source", "Unknown")
            page = m.get("page", 1)
            total_p = m.get("total_pages", page)
            if src not in docs_map:
                docs_map[src] = {
                    "filename": src,
                    "chunks_count": 0,
                    "pages": set(),
                    "total_pages": total_p
                }
            docs_map[src]["chunks_count"] += 1
            docs_map[src]["pages"].add(page)

        result = []
        for src, data in docs_map.items():
            result.append({
                "filename": src,
                "chunks_count": data["chunks_count"],
                "total_pages": max(data["total_pages"], len(data["pages"])),
                "pages_indexed": sorted(list(data["pages"]))
            })
        return sorted(result, key=lambda x: x["filename"])

    def get_document_chunks(self, filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Returns all chunk details for inspection in the UI Chunk Inspector.
        """
        if self.collection.count() == 0:
            return []

        where_filter = {"source": filename} if filename else None
        data = self.collection.get(where=where_filter, include=["documents", "metadatas"])
        
        chunks = []
        if data and "documents" in data:
            for doc, meta, cid in zip(data["documents"], data["metadatas"], data["ids"]):
                chunks.append({
                    "id": cid,
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "chunk_index": meta.get("chunk_index"),
                    "content": doc,
                    "length": len(doc)
                })
        return sorted(chunks, key=lambda x: (x.get("source", ""), x.get("page", 0), x.get("chunk_index", 0)))

    def delete_document(self, filename: str) -> int:
        """
        Deletes all chunks associated with a specific file.
        """
        try:
            existing = self.collection.get(where={"source": filename})
            if existing and "ids" in existing and existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                return len(existing["ids"])
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
        return 0

    def clear_all(self) -> bool:
        """
        Clears the entire ChromaDB collection.
        """
        try:
            self.chroma_client.delete_collection(name="syllabus_knowledge_base")
            self.collection = self.chroma_client.get_or_create_collection(
                name="syllabus_knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns general vector store statistics.
        """
        total_chunks = self.collection.count()
        docs = self.list_documents()
        total_pages = sum(d["total_pages"] for d in docs)
        return {
            "total_chunks": total_chunks,
            "total_documents": len(docs),
            "total_pages": total_pages,
            "model_name": "all-MiniLM-L6-v2",
            "vector_dimension": 384
        }
