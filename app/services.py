import os
import duckdb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, Language as LCLanguage
except ImportError:  # older langchain layout
    from langchain.text_splitter import RecursiveCharacterTextSplitter, Language as LCLanguage
try:
    from llama_index.core.node_parser import CodeSplitter
    CODESPLITTER_AVAILABLE = True
except ImportError:
    try:
        from llama_index.node_parser import CodeSplitter
        CODESPLITTER_AVAILABLE = True
    except ImportError:
        CODESPLITTER_AVAILABLE = False
        CodeSplitter = None
from llama_index.core.schema import Document
import logging
import numpy as np, json
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from pathlib import Path
try:
    import yaml  # for OKF YAML frontmatter parsing
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None
try:
    from tree_sitter import Parser, Language
    from tree_sitter_language_pack import get_language
    TREE_SITTER_AVAILABLE = True
except ImportError:
    try:
        from tree_sitter import Parser, Language  
        from tree_sitter_languages import get_language
        TREE_SITTER_AVAILABLE = True
    except ImportError:
        TREE_SITTER_AVAILABLE = False
        Parser = None
        get_language = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
# Primary model: high-quality multilingual embeddings (768 dimensions)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Fallback model: faster, smaller, but good quality (384 dimensions)  
FALLBACK_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "data/rag.duckdb"
UPLOADS_DIR = "uploads"
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- Configurable Parameters ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# --- Global Variables ---
model = None
reranker = None
db_connection = None

# --- Initialization ---
def initialize_services():
    """Initializes the models and database connection."""
    global model, db_connection, reranker
    
    logger.info("Initializing services...")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # Load the embedding model with fallback
    if model is None:
        try:
            logger.info(f"Loading sentence transformer model: {MODEL_NAME}")
            model = SentenceTransformer(MODEL_NAME)
            logger.info("Primary model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load primary model {MODEL_NAME}: {e}")
            logger.info(f"Trying fallback model: {FALLBACK_MODEL}")
            try:
                model = SentenceTransformer(FALLBACK_MODEL)
                logger.info("Fallback model loaded successfully.")
            except Exception as e2:
                logger.error(f"Failed to load fallback model {FALLBACK_MODEL}: {e2}")
                raise RuntimeError(f"Could not load any embedding model. Primary error: {e}, Fallback error: {e2}")

    # Load the reranker model
    if reranker is None:
        try:
            logger.info(f"Loading reranker model: {RERANKER_MODEL_NAME}")
            reranker = CrossEncoder(RERANKER_MODEL_NAME)
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load reranker model {RERANKER_MODEL_NAME}: {e}. Reranking will be disabled.")
            reranker = None

    # Get embedding dimension from the loaded model
    if hasattr(model, 'get_sentence_embedding_dimension'):
        embedding_dim = model.get_sentence_embedding_dimension()
    else:
        # Fallback: create a test embedding to determine dimension
        test_embedding = model.encode("test")
        embedding_dim = len(test_embedding)
    
    logger.info(f"Detected embedding dimension: {embedding_dim}")

    # Initialize DuckDB
    if db_connection is None:
        logger.info(f"Initializing DuckDB at {DB_PATH}")
        db_connection = duckdb.connect(database=DB_PATH, read_only=False)
        
        # Install and load VSS and FTS extensions
        db_connection.execute("INSTALL vss; LOAD vss;")
        db_connection.execute("INSTALL fts; LOAD fts;")
        
        # Create table for chunks if it doesn't exist
        # Dynamic embedding vector size based on the model
        db_connection.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id UBIGINT PRIMARY KEY,
                file_name VARCHAR,
                chunk_index INTEGER,
                content TEXT,
                metadata JSON,
                embedding FLOAT[{embedding_dim}]
            );
        """)
        
        # Create FTS index on content for keyword search
        try:
            db_connection.execute("PRAGMA create_fts_index('chunks', 'id', 'content', overwrite=1);")
            logger.info("Successfully created FTS index on 'chunks.content'.")
        except Exception as e:
            logger.error(f"Failed to create FTS index: {e}")
        # Add metadata column if it doesn't exist for backward compatibility
        try:
            db_connection.execute("ALTER TABLE chunks ADD COLUMN metadata JSON;")
            logger.info("Added 'metadata' column to 'chunks' table.")
        except (getattr(duckdb, "CatalogException", Exception),):
            # Column already exists, which is fine
            pass
        logger.info(f"DuckDB initialized and table 'chunks' is ready with embedding dimension {embedding_dim}.")

# --- File Processing ---
def _extract_text_from_txt(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def _extract_text_from_md(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def _extract_text_from_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_file(filepath: str) -> str:
    """Extracts text from a file based on its extension."""
    ext = os.path.splitext(filepath)[1].lower()
    
    # All supported text-based file extensions  
    # Includes programming languages, markup, config, and data formats
    supported_text_extensions = {
        # Programming languages
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc', '.cxx', 
        '.cs', '.go', '.rs', '.php', '.rb', '.scala', '.swift',
        # Shell scripts
        '.sh', '.bash', '.zsh', '.fish',
        # Web technologies
        '.html', '.htm', '.css', '.scss', '.sass', '.xml',
        # Database
        '.sql',
        # Data formats
        '.json', '.yaml', '.yml', '.toml', '.csv',
        # Configuration
        '.ini', '.conf', '.config', '.env',
        # Documentation
        '.md', '.txt', '.rst', '.asciidoc',
        # Other text files
        '.log', '.gitignore', '.dockerfile'
    }
    
    if ext == '.pdf':
        return _extract_text_from_pdf(filepath)
    elif ext in supported_text_extensions:
        # All these can be read as plain text
        return _extract_text_from_txt(filepath)
    else:
        logger.warning(f"Unsupported file type: {ext}. Skipping.")
        return ""

def _extract_ast_chunks(filepath: str, text: str, lang: str) -> List[Dict[str, Any]]:
    """Extracts code chunks using tree-sitter for proper AST parsing."""
    if not TREE_SITTER_AVAILABLE:
        logger.warning("Tree-sitter not available for AST parsing.")
        return []
        
    file_name = os.path.basename(filepath)
    try:
        language = get_language(lang)
        parser = Parser()
        parser.language = language  # Исправлен API tree-sitter
    except Exception as e:
        logger.error(f"Could not get tree-sitter language for '{lang}' for file {file_name}: {e}")
        return []

    tree = parser.parse(bytes(text, "utf8"))
    
    queries = {
        'python': "(function_definition) @func (class_definition) @class",
        'javascript': "(function_declaration) @func (class_declaration) @class (method_definition) @func",
        'typescript': "(function_declaration) @func (class_declaration) @class (method_definition) @func (interface_declaration) @class",
        'java': "(method_declaration) @func (class_declaration) @class (interface_declaration) @class",
        'cpp': "(function_definition) @func (class_specifier) @class (struct_specifier) @class",
        'c': "(function_definition) @func (struct_specifier) @class",
        'rust': "(function_item) @func (struct_item) @class (trait_item) @class",
        'go': "(function_declaration) @func (type_spec (struct_type)) @class"
    }
    query_string = queries.get(lang)
    if not query_string: return []

    query = language.query(query_string)
    captures = query.captures(tree.root_node)
    
    chunks = []
    for capture in captures:
        if not isinstance(capture, tuple) or len(capture) != 2:
            logger.warning(f"Skipping malformed tree-sitter capture: {capture}")
            continue
        node, capture_name = capture

        identifier_node = node.child_by_field_name("name") or node.child_by_field_name("declarator")
        identifier = identifier_node.text.decode('utf8') if identifier_node else "anonymous"

        chunks.append({
            "text": node.text.decode('utf8'),
            "metadata": {
                "file_name": file_name, "type": "code_entity",
                "entity_type": "function" if capture_name == 'func' else "class",
                "language": lang, "identifier": identifier,
                "start_line": node.start_point[0] + 1, "end_line": node.end_point[0] + 1,
            }
        })
    return chunks

# --- Chunking ---
def get_chunks_for_file(filepath: str, text: str) -> List[Dict[str, Any]]:
    """
    Selects a chunking strategy based on file type and returns a list of dictionaries,
    each containing the chunk text and its metadata.
    """
    ext = Path(filepath).suffix.lower()
    file_name = os.path.basename(filepath)

    # Map extensions to LlamaIndex CodeSplitter languages
    # Comprehensive mapping for all supported tree-sitter languages
    lang_map = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.cs': 'c_sharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.scala': 'scala',
        '.swift': 'swift',
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
        '.scss': 'css',
        '.sass': 'css',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'bash',
        '.fish': 'bash',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.xml': 'xml',
        '.sql': 'sql',
    }

    chunks = []

    if ext in lang_map:
        # --- Try AST-based chunking first ---
        try:
            ast_chunks = _extract_ast_chunks(filepath, text, lang_map[ext])
            if ast_chunks:
                logger.info(f"Used AST parser for {file_name}, found {len(ast_chunks)} entities.")
                return ast_chunks
            else:
                logger.info(f"AST parser found no entities for {file_name}, falling back.")
        except Exception as e:
            logger.warning(f"AST parsing failed for {file_name}: {e}. Falling back.")

        # --- Fallback to LlamaIndex CodeSplitter ---
        if CODESPLITTER_AVAILABLE:
            try:
                splitter = CodeSplitter(
                    language=lang_map[ext],
                    chunk_lines=40,  # Corresponds roughly to CHUNK_SIZE
                    chunk_lines_overlap=10, # Corresponds roughly to CHUNK_OVERLAP
                    max_chars=CHUNK_SIZE,
                )
                nodes = splitter.get_nodes_from_documents([Document(text=text)])
                for node in nodes:
                    metadata = node.metadata or {}
                    metadata.update({"file_name": file_name, "type": "code", "language": lang_map[ext]})
                    chunks.append({"text": node.get_content(), "metadata": metadata})
                logger.info(f"Used LlamaIndex CodeSplitter for {file_name}")
                return chunks
            except Exception as e:
                logger.warning(f"LlamaIndex CodeSplitter failed for {file_name}: {e}. Falling back to default.")
        else:
            logger.warning(f"CodeSplitter not available for {file_name}. Please install tree-sitter-language-pack. Falling back to default.")

    # For Markdown, use LangChain's splitter
    if ext == '.md':
        lc_splitter = RecursiveCharacterTextSplitter.from_language(
            language=LCLanguage.MARKDOWN,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        text_chunks = lc_splitter.split_text(text)
        for chunk_text in text_chunks:
            chunks.append({"text": chunk_text, "metadata": {"file_name": file_name, "type": "markdown"}})
        return chunks

    # For config files, split by '---' or treat as whole
    if ext in ['.json', '.yaml', '.yml', '.ini', '.toml']:
        text_chunks = [part for part in text.split('\n---\n') if part.strip()] if '---' in text else [text]
        for chunk_text in text_chunks:
            chunks.append({"text": chunk_text, "metadata": {"file_name": file_name, "type": "config"}})
        return chunks

    # Default splitter for .txt, .pdf, and other text-like files
    lc_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    text_chunks = lc_splitter.split_text(text)
    for chunk_text in text_chunks:
        chunks.append({"text": chunk_text, "metadata": {"file_name": file_name, "type": "text"}})
    return chunks

# --- Database Querying ---
def get_total_chunks_count() -> int:
    """Returns the total number of chunks in the database."""
    if db_connection is None:
        return 0
    try:
        count_result = db_connection.execute("SELECT COUNT(id) FROM chunks").fetchone()
        return count_result[0] if count_result else 0
    except Exception as e:
        logger.error(f"Could not get chunk count: {e}")
        return 0

def search_chunks(query: str, top_k: int = 5, search_type: str = 'hybrid',
                  expand_query: bool = False, use_reranker: bool = True,
                  filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Performs search with multiple strategies: semantic, keyword (BM25), or hybrid.
    Includes optional query expansion and reranking.
    """
    if model is None or db_connection is None:
        initialize_services()

    if not query or not query.strip():
        return []

    # Build optional metadata filter (OKF: type / project / report_type / tags / report_date range).
    _fclauses, filter_params = [], []
    for _key, _jpath in (("project", "$.project"), ("report_type", "$.report_type"), ("type", "$.type")):
        if filters and filters.get(_key):
            _fclauses.append(f"json_extract_string(metadata, '{_jpath}') = ?"); filter_params.append(filters[_key])
    if filters and filters.get("date_from"):
        _fclauses.append("json_extract_string(metadata, '$.report_date') >= ?"); filter_params.append(filters["date_from"])
    if filters and filters.get("date_to"):
        _fclauses.append("json_extract_string(metadata, '$.report_date') <= ?"); filter_params.append(filters["date_to"])
    if filters and filters.get("tags"):
        _tags = filters["tags"]
        if isinstance(_tags, str):
            _tags = [t.strip() for t in _tags.split(",") if t.strip()]
        _tag_or = []
        for _t in _tags:
            # OKF tags are stored as a JSON array; match membership via json_contains.
            _tag_or.append("json_contains(json_extract(metadata, '$.tags'), ?)")
            filter_params.append(json.dumps(_t))
        if _tag_or:
            _fclauses.append("(" + " OR ".join(_tag_or) + ")")
    filter_sql = " AND ".join(_fclauses)

    logger.info(f"Performing {search_type} search for query: '{query}' with top_k={top_k}, expand_query={expand_query}, use_reranker={use_reranker}, filters={filters}")

    # 1. Query Expansion (optional)
    if expand_query:
        try:
            logger.info("Performing query expansion...")
            initial_results = search_chunks(query, top_k=2, search_type='semantic', expand_query=False, use_reranker=False, filters=filters)
            
            expanded_keywords = set()
            for res in initial_results:
                if res.get('metadata') and 'keywords' in res['metadata']:
                    expanded_keywords.update(res['metadata']['keywords'])
            
            if expanded_keywords:
                original_query_words = set(query.lower().split())
                new_keywords = [kw for kw in expanded_keywords if kw not in original_query_words]
                if new_keywords:
                    expanded_query_str = " ".join(new_keywords)
                    query = f"{query} {expanded_query_str}"
                    logger.info(f"Expanded query: '{query}'")
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Continuing with original query.")

    # Determine how many results to fetch for reranking
    fetch_k = 50 if use_reranker and reranker else top_k

    # 2. Fetch results from semantic and/or keyword search
    semantic_results, keyword_results = [], []

    if search_type in ['semantic', 'hybrid']:
        try:
            query_embedding = model.encode(query).astype('float32').tolist()
            embedding_dim = len(query_embedding)
            _where = f"WHERE {filter_sql}" if filter_sql else ""
            res = db_connection.execute(
                f"SELECT id, array_cosine_similarity(embedding, ?::FLOAT[{embedding_dim}]) AS score FROM chunks {_where} ORDER BY score DESC LIMIT ?",
                (query_embedding, *filter_params, fetch_k)
            ).fetchall()
            semantic_results = [{"id": row[0], "score": row[1]} for row in res]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")

    if search_type in ['keyword', 'hybrid']:
        try:
            _and = f"AND {filter_sql}" if filter_sql else ""
            res = db_connection.execute(
                f"SELECT id, fts_main_chunks.match_bm25(id, ?) AS score FROM chunks WHERE score IS NOT NULL {_and} ORDER BY score DESC LIMIT ?",
                (query, *filter_params, fetch_k)
            ).fetchall()
            keyword_results = [{"id": row[0], "score": row[1]} for row in res]
        except Exception as e:
            logger.error(f"Keyword search (FTS) failed: {e}")

    # 3. Combine results (Hybrid Search using RRF)
    combined_results = {}
    if search_type == 'hybrid':
        k_rrf = 60
        for rank, doc in enumerate(semantic_results):
            combined_results[doc['id']] = combined_results.get(doc['id'], 0) + 1 / (k_rrf + rank)
        for rank, doc in enumerate(keyword_results):
            combined_results[doc['id']] = combined_results.get(doc['id'], 0) + 1 / (k_rrf + rank)
        sorted_ids = [k for k, v in sorted(combined_results.items(), key=lambda item: item[1], reverse=True)]
    elif search_type == 'semantic':
        sorted_ids = [res['id'] for res in semantic_results]
    else:
        sorted_ids = [res['id'] for res in keyword_results]

    if not sorted_ids:
        return []

    # Fetch full chunk data
    placeholders = ','.join(['?'] * len(sorted_ids))
    all_chunks_data = db_connection.execute(
        f"SELECT id, file_name, content, metadata FROM chunks WHERE id IN ({placeholders})",
        sorted_ids
    ).fetchall()
    chunks_map = {row[0]: {"id": row[0], "file_name": row[1], "content": row[2], "metadata": json.loads(row[3]) if row[3] else {}} for row in all_chunks_data}
    ordered_chunks = [chunks_map[id] for id in sorted_ids if id in chunks_map]

    # 4. Rerank (optional)
    if use_reranker and reranker and search_type != 'keyword':
        logger.info(f"Reranking top {len(ordered_chunks)} results...")
        rerank_pairs = [[query, chunk['content']] for chunk in ordered_chunks]
        if rerank_pairs:
            try:
                scores = reranker.predict(rerank_pairs, show_progress_bar=False)
                for i, chunk in enumerate(ordered_chunks):
                    chunk['rerank_score'] = scores[i]
                ordered_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
            except Exception as e:
                logger.error(f"Reranking failed: {e}")

    # 5. Format final output
    final_results = []
    semantic_scores = {res['id']: res['score'] for res in semantic_results}
    keyword_scores = {res['id']: res['score'] for res in keyword_results}

    for chunk in ordered_chunks[:top_k]:
        score, score_type = 0, 'N/A'
        if 'rerank_score' in chunk:
            score, score_type = chunk['rerank_score'], 'rerank_score'
        elif search_type == 'hybrid':
            score, score_type = combined_results.get(chunk['id'], 0.0), 'rrf_score'
        elif search_type == 'semantic':
            score, score_type = semantic_scores.get(chunk['id'], 0.0), 'cosine_similarity'
        elif search_type == 'keyword':
            score, score_type = keyword_scores.get(chunk['id'], 0.0), 'bm25_score'

        final_results.append({
            "file_name": chunk["file_name"], "content": chunk["content"],
            "metadata": chunk["metadata"], "score": f"{score:.4f}", "score_type": score_type
        })

    return final_results

def process_single_file(filepath: str, use_tfidf_keywords: bool = False, extra_metadata: Dict[str, Any] = None,
                        text_override: str = None) -> List[Dict[str, Any]]:
    """
    Processes a single file from its path, generates embeddings, and stores it in DuckDB.
    This is an atomic version for single uploads. If text_override is given, it is used
    as the document body instead of re-reading the file (used to feed frontmatter-stripped
    report bodies); filepath is still used for naming and chunk-language detection.
    """
    if model is None or db_connection is None:
        initialize_services()

    file_name = os.path.basename(filepath)
    logger.info(f"Processing single file: {file_name}")

    try:
        text = text_override if text_override is not None else extract_text_from_file(filepath)
        if not text or not text.strip():
            logger.warning(f"No text extracted from {file_name} or file is empty. Skipping.")
            return []

        chunks_with_meta = get_chunks_for_file(filepath, text)
        if not chunks_with_meta:
            logger.warning(f"No chunks created for {file_name}. Skipping.")
            return []

        # Merge caller-supplied metadata (e.g. report {project, report_type, ...}) into every chunk.
        if extra_metadata:
            for _c in chunks_with_meta:
                _c['metadata'].update(extra_metadata)

        # Note: TF-IDF is corpus-based. For a single file, this is just term frequency.
        # It's disabled by default for this function.
        if use_tfidf_keywords:
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=10, ngram_range=(1, 2))
                vectorizer.fit([text])
                keywords = vectorizer.get_feature_names_out().tolist()
                for chunk in chunks_with_meta:
                    chunk['metadata']['keywords'] = keywords
            except Exception as e:
                logger.warning(f"Could not generate keywords for single file {file_name}: {e}")

        logger.info(f"Generated {len(chunks_with_meta)} chunks for {file_name}.")

        chunk_contents = [c['text'] for c in chunks_with_meta]
        embeddings = model.encode(chunk_contents, show_progress_bar=False)
        
        max_id_result = db_connection.execute("SELECT MAX(id) FROM chunks").fetchone()
        current_max_id = max_id_result[0] if max_id_result and max_id_result[0] is not None else 0

        processed_chunks_json = []
        for i, (chunk_data, embedding) in enumerate(zip(chunks_with_meta, embeddings)):
            chunk_id = current_max_id + i + 1
            embedding_float = embedding.astype('float32').tolist()
            metadata_json = json.dumps(chunk_data['metadata'])

            db_connection.execute(
                "INSERT INTO chunks (id, file_name, chunk_index, content, metadata, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                (chunk_id, file_name, i, chunk_data['text'], metadata_json, embedding_float)
            )
            
            processed_chunks_json.append(chunk_data)
        
        logger.info(f"Successfully inserted {len(chunks_with_meta)} chunks for {file_name} into DuckDB.")
        return processed_chunks_json

    except Exception as e:
        logger.error(f"Failed to process single file {filepath}: {e}", exc_info=True)
        raise

# --- Report ingestion (Engram reports hub, OKF-aligned) ---
# Recommended report_type vocabulary (free-form; any folder name is accepted).
REPORT_TYPES = {"daily", "weekly", "analysis", "planning", "implementation",
                "premortem", "reference", "archive"}
# OKF reserved filenames — navigation/history, not concept documents (skip on ingest).
RESERVED_FILES = {"index.md", "log.md"}
TEXT_EXTS = {".md", ".markdown", ".txt", ".mdx"}


def _parse_frontmatter(raw: str):
    """Parse a leading OKF/YAML frontmatter block. Returns (meta: dict, body: str)."""
    if not YAML_AVAILABLE or not raw.startswith("---"):
        return {}, raw
    import re
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, re.DOTALL)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}, raw
    if not isinstance(meta, dict):
        return {}, raw
    return meta, m.group(2)


def _derive_project_type(abspath: str):
    """Derive (project, report_type) from the path convention
    .../<project>/<report_type>/file or .../projects/<project>/<report_type>/file.
    report_type is the directory directly under the project dir (None if file sits there)."""
    parts = list(Path(abspath).parts)
    lower = [p.lower() for p in parts]
    project = rtype = None
    anchor = None
    for key in ("projects", "global"):
        if key in lower:
            anchor = lower.index(key)
            break
    if anchor is not None:
        if key == "global":
            project = "global"
            base = anchor  # report_type is the dir after 'global'
        else:
            if anchor + 1 <= len(parts) - 1:
                project = parts[anchor + 1]
            base = anchor + 1  # report_type is the dir after the project dir
        # report_type = next segment, only if it's a directory (not the filename itself)
        if base + 1 <= len(parts) - 2:
            rtype = parts[base + 1]
    return project, rtype


def ingest_report(path: str, project: str = None, report_type: str = None,
                  report_date: str = None, title: str = None,
                  description: str = None, tags=None, doc_type: str = None) -> Dict[str, Any]:
    """
    Ingest a single report as an OKF concept document. Metadata precedence:
      explicit arg > YAML frontmatter > path convention > filename/mtime fallback.
    Stored keys: type (OKF, default 'Report'), title, description, resource, tags[],
    timestamp; plus Engram report profile: project, report_type, report_date, source_path.
    """
    import datetime, re
    abspath = os.path.abspath(path)
    if not os.path.isfile(abspath):
        raise FileNotFoundError(abspath)
    if os.path.basename(abspath).lower() in RESERVED_FILES:
        return {"source_path": abspath, "chunks_added": 0, "skipped": "reserved-file", "metadata": {}}

    # Read raw text + split OKF frontmatter (text formats only; binaries extracted later).
    ext = Path(abspath).suffix.lower()
    fm, body, text_override = {}, None, None
    if ext in TEXT_EXTS:
        try:
            raw = open(abspath, encoding="utf-8", errors="replace").read()
        except Exception:
            raw = ""
        fm, body = _parse_frontmatter(raw)

    path_project, path_rtype = _derive_project_type(abspath)

    project = project or fm.get("project") or path_project
    if not project:
        raise ValueError(
            "project not provided and not derivable from path "
            "(expected .../projects/<project>/... or .../global/...)")
    report_type = report_type or fm.get("report_type") or path_rtype
    doc_type = doc_type or fm.get("type") or "Report"
    title = title or fm.get("title") or os.path.splitext(os.path.basename(abspath))[0]
    description = description or fm.get("description")
    tags = tags if tags is not None else fm.get("tags")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    if not isinstance(tags, list):
        tags = []
    resource = fm.get("resource") or f"file://{abspath}"
    timestamp = fm.get("timestamp") or datetime.datetime.fromtimestamp(
        os.path.getmtime(abspath)).isoformat(timespec="seconds")

    if not report_date:
        report_date = fm.get("report_date")
    if not report_date:
        base = os.path.basename(abspath)
        m = re.search(r"(\d{4}-\d{2}-\d{2})", base)
        if m:
            report_date = m.group(1)
        else:
            m8 = re.search(r"(\d{4})(\d{2})(\d{2})", base)  # e.g. 20251008
            report_date = (f"{m8.group(1)}-{m8.group(2)}-{m8.group(3)}" if m8
                           else datetime.date.fromtimestamp(os.path.getmtime(abspath)).isoformat())

    meta = {
        "type": doc_type,
        "project": project,
        "report_type": report_type,
        "report_date": report_date,
        "title": title,
        "description": description,
        "tags": tags,
        "timestamp": timestamp,
        "resource": resource,
        "source_path": abspath,
    }

    # YAML auto-parses unquoted dates/timestamps to date/datetime objects, which are not
    # JSON-serializable — coerce them (and any in lists) to ISO strings so metadata stores cleanly.
    def _jsonable(v):
        if isinstance(v, (datetime.date, datetime.datetime)):
            return v.isoformat()
        if isinstance(v, list):
            return [_jsonable(x) for x in v]
        return v
    meta = {k: _jsonable(v) for k, v in meta.items() if v is not None}

    # For text reports, embed the frontmatter-stripped body and prepend title+description
    # so the leading chunk is retrievable by its own summary.
    if body is not None:
        head = f"# {title}\n" + (f"{description}\n\n" if description else "\n")
        text_override = head + body

    # Upsert: drop any existing chunks for this source_path so re-ingest is idempotent
    # (a sync re-ingesting a changed file won't duplicate chunks). No-op on first ingest.
    replaced = 0
    if db_connection is None:
        initialize_services()
    try:
        replaced = db_connection.execute(
            "SELECT count(*) FROM chunks WHERE json_extract_string(metadata, '$.source_path') = ?",
            (abspath,)).fetchone()[0]
        if replaced:
            db_connection.execute(
                "DELETE FROM chunks WHERE json_extract_string(metadata, '$.source_path') = ?",
                (abspath,))
    except Exception as e:
        logger.warning(f"Upsert delete failed for {abspath}: {e}")

    chunks = process_single_file(abspath, use_tfidf_keywords=False,
                                 extra_metadata=meta, text_override=text_override)
    return {"source_path": abspath, "chunks_added": len(chunks),
            "replaced_chunks": replaced, "metadata": meta}


def delete_report(source_path: str) -> int:
    """Delete all chunks for a given source_path (the path used at ingest, e.g.
    /reports/...). Returns the number of chunks removed. Used by the sync watcher to
    propagate file deletions into the index."""
    if db_connection is None:
        initialize_services()
    n = db_connection.execute(
        "SELECT count(*) FROM chunks WHERE json_extract_string(metadata, '$.source_path') = ?",
        (source_path,)).fetchone()[0]
    if n:
        db_connection.execute(
            "DELETE FROM chunks WHERE json_extract_string(metadata, '$.source_path') = ?",
            (source_path,))
    logger.info(f"delete_report: removed {n} chunks for {source_path}")
    return n

# --- Main Processing Logic ---
def process_and_embed_files(use_tfidf_keywords: bool = True, top_n_keywords: int = 5) -> List[Dict[str, Any]]:
    """
    Processes all supported files in the uploads directory,
    generates embeddings, and stores them in DuckDB.
    Optionally adds TF-IDF based keywords to metadata.
    """
    try:
        if model is None or db_connection is None:
            initialize_services()

        processed_chunks_json = []
        
        # Get initial list of files from the uploads directory
        all_files_in_dir = [os.path.join(UPLOADS_DIR, f) for f in os.listdir(UPLOADS_DIR) if os.path.isfile(os.path.join(UPLOADS_DIR, f))]

        # Filter out empty/unreadable files, build a dictionary of file texts
        file_texts = {}
        total_files_size = 0
        
        logger.info(f"Reading {len(all_files_in_dir)} files into memory...")
        
        for i, filepath in enumerate(all_files_in_dir):
            file_name = os.path.basename(filepath)
            logger.info(f"Reading file {i+1}/{len(all_files_in_dir)}: {file_name}")
            
            try:
                text = extract_text_from_file(filepath)
                if text and text.strip():
                    file_size = len(text)
                    
                    total_files_size += file_size
                    logger.info(f"  ✅ {file_name}: {file_size} chars ({file_size/1024:.1f} KB)")
                    
                    file_texts[filepath] = text
                else:
                    logger.warning(f"  ⚠️ No text extracted from {file_name} or file is empty. Skipping and removing.")
                    try:
                        os.remove(filepath)
                    except OSError as e:
                        logger.error(f"Error removing empty/unreadable file {filepath}: {e}")
            except Exception as e:
                logger.error(f"❌ Error reading file {file_name}: {e}", exc_info=True)
                try:
                    os.remove(filepath)
                except OSError:
                    pass

        files_to_process = list(file_texts.keys())
        logger.info(f"Successfully loaded {len(files_to_process)} files, total size: {total_files_size/1024/1024:.2f} MB")
        
        if not files_to_process:
            return []

        # --- TF-IDF Keyword Extraction ---
        file_keywords = {}
        if use_tfidf_keywords:
            logger.info("Calculating TF-IDF keywords for uploaded files...")
            
            # Используем все файлы для TF-IDF без ограничений по размеру
            tfidf_files = []
            tfidf_texts = []
            
            for filepath in files_to_process:
                file_name = os.path.basename(filepath)
                text = file_texts[filepath]
                file_size = len(text)
                
                tfidf_files.append(filepath)
                tfidf_texts.append(text)
            
            total_corpus_size = sum(len(text) for text in tfidf_texts)
            logger.info(f"TF-IDF corpus: {len(tfidf_files)} files, {total_corpus_size/1024/1024:.2f} MB")
            
            # Обрабатываем TF-IDF если есть файлы
            if len(tfidf_texts) > 0:
                try:
                    logger.info(f"Creating TfidfVectorizer for {len(tfidf_texts)} files...")
                    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
                    
                    logger.info("Fitting TF-IDF matrix...")
                    tfidf_matrix = vectorizer.fit_transform(tfidf_texts)
                    
                    logger.info("Getting feature names...")
                    feature_names = vectorizer.get_feature_names_out()
                    
                    logger.info(f"Extracting keywords for {len(tfidf_files)} files...")
                    for i, filepath in enumerate(tfidf_files):
                        scores = tfidf_matrix[i].toarray().flatten()
                        top_keyword_indices = scores.argsort()[-top_n_keywords:][::-1]
                        keywords = [feature_names[idx] for idx in top_keyword_indices if scores[idx] > 0.05]
                        file_keywords[filepath] = keywords
                        
                    logger.info(f"TF-IDF extraction completed successfully for {len(file_keywords)} files")
                    
                    # Для больших файлов добавляем пустой список ключевых слов
                    for filepath in files_to_process:
                        if filepath not in file_keywords:
                            file_keywords[filepath] = []  # Пустой список для больших файлов
                            
                except MemoryError as e:
                    logger.error(f"❌ Memory error during TF-IDF: {e}. Skipping TF-IDF keywords.")
                except Exception as e:
                    logger.error(f"❌ Failed to generate TF-IDF keywords: {e}", exc_info=True)

        for filepath in files_to_process:
            file_name = os.path.basename(filepath)
            logger.info(f"Processing file: {file_name}")

            try:
                text = file_texts[filepath]  # Text is already validated and read

                chunks_with_meta = get_chunks_for_file(filepath, text)
                if not chunks_with_meta:
                    logger.warning(f"No chunks created for {file_name}. Skipping.")
                    os.remove(filepath)
                    continue
                
                # Add TF-IDF keywords to metadata
                if filepath in file_keywords:
                    for chunk in chunks_with_meta:
                        chunk['metadata']['keywords'] = file_keywords[filepath]

                logger.info(f"Generated {len(chunks_with_meta)} chunks for {file_name}.")

                # Extract just the text for embedding
                chunk_contents = [c['text'] for c in chunks_with_meta]
                embeddings = model.encode(chunk_contents, show_progress_bar=True)
                
                max_id_result = db_connection.execute("SELECT MAX(id) FROM chunks").fetchone()
                current_max_id = max_id_result[0] if max_id_result and max_id_result[0] is not None else 0

                for i, (chunk_data, embedding) in enumerate(zip(chunks_with_meta, embeddings)):
                    chunk_id = current_max_id + i + 1
                    embedding_float = embedding.astype('float32').tolist()
                    
                    # Convert metadata dict to JSON string for DB
                    metadata_json = json.dumps(chunk_data['metadata'])

                    db_connection.execute(
                        "INSERT INTO chunks (id, file_name, chunk_index, content, metadata, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                        (chunk_id, file_name, i, chunk_data['text'], metadata_json, embedding_float)
                    )
                    
                    processed_chunks_json.append({
                        "id": int(chunk_id),
                        "file_name": file_name,
                        "chunk_index": i,
                        "content": chunk_data['text'],
                        "metadata": chunk_data['metadata'],
                    })
                
                logger.info(f"Successfully inserted {len(chunks_with_meta)} chunks for {file_name} into DuckDB.")
                
                os.remove(filepath)
                logger.info(f"Removed processed file: {filepath}")

            except Exception as e:
                logger.error(f"Failed to process file {filepath}: {e}", exc_info=True)

        return processed_chunks_json
        
    except MemoryError as e:
        logger.error(f"❌ CRITICAL: Out of memory during file processing: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ CRITICAL: Unexpected error during file processing: {e}", exc_info=True)
        return []

def get_uploaded_files() -> List[str]:
    """Returns a list of filenames in the uploads directory."""
    if not os.path.exists(UPLOADS_DIR):
        return []
    return [f for f in os.listdir(UPLOADS_DIR) if os.path.isfile(os.path.join(UPLOADS_DIR, f))] 