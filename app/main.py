import os
import shutil
import json
import logging
from fastapi import FastAPI, File, UploadFile, Request, Form, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Optional

from . import services

# Setup logging
logger = logging.getLogger(__name__)

def get_default_extensions():
    """Returns the default set of allowed file extensions."""
    return {
        ".txt", ".md", ".pdf", ".ini", ".toml",  # Text and config files
        ".py", ".js", ".ts", ".jsx", ".tsx",     # Python and JavaScript/TypeScript
        ".java", ".c", ".cpp", ".cc", ".cxx",    # Java and C/C++
        ".cs", ".go", ".rs", ".php", ".rb",      # C#, Go, Rust, PHP, Ruby
        ".scala", ".swift", ".html", ".htm",     # Scala, Swift, HTML
        ".css", ".scss", ".sass",                # CSS and preprocessors
        ".sh", ".bash", ".zsh", ".fish",         # Shell scripts
        ".json", ".yaml", ".yml", ".xml", ".sql" # Data and query languages
    }

app = FastAPI(
    title="Python RAG Server",
    description="A server for document processing, embedding, and storage in DuckDB.",
    version="1.0.0"
)

# --- Setup ---
templates = Jinja2Templates(directory="templates")
# This is for if we add any CSS/JS files later
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """On startup, initialize the services (model, DB)."""
    services.initialize_services()

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main page."""
    uploaded_files = services.get_uploaded_files()
    total_chunks = services.get_total_chunks_count()
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "uploaded_files": uploaded_files,
        "processed_chunks": None,
        "total_chunks": total_chunks
    })

@app.post("/upload-files/", response_class=HTMLResponse)
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    """Handles file uploads and saves them to the 'uploads' directory."""
    for file in files:
        # Security: only allow specific extensions
        # Comprehensive list including all supported programming languages
        allowed_extensions = {
            ".txt", ".md", ".pdf", ".ini", ".toml",  # Text and config files
            ".py", ".js", ".ts", ".jsx", ".tsx",     # Python and JavaScript/TypeScript
            ".java", ".c", ".cpp", ".cc", ".cxx",    # Java and C/C++
            ".cs", ".go", ".rs", ".php", ".rb",      # C#, Go, Rust, PHP, Ruby
            ".scala", ".swift", ".html", ".htm",     # Scala, Swift, HTML
            ".css", ".scss", ".sass",                # CSS and preprocessors
            ".sh", ".bash", ".zsh", ".fish",         # Shell scripts
            ".json", ".yaml", ".yml", ".xml", ".sql" # Data and query languages
        }
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            # In a real app, you'd return a proper error response
            continue

        content = await file.read()
        if not content:
            continue  # Skip empty files

        sanitized_filename = os.path.basename(file.filename)
        file_path = os.path.join(services.UPLOADS_DIR, sanitized_filename)
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    
    uploaded_files = services.get_uploaded_files()
    total_chunks = services.get_total_chunks_count()
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "uploaded_files": uploaded_files,
        "processed_chunks": None,
        "total_chunks": total_chunks
    })

@app.post("/upload-directory/", response_class=HTMLResponse)
async def upload_directory(request: Request):
    """Handles directory uploads by saving all relevant files into the 'uploads' directory."""
    try:
        # Получаем форму напрямую из request
        form = await request.form()
        
        # Отладочное логирование - что в форме
        logger.info(f"Form keys: {list(form.keys())}")
        for key in form.keys():
            value = form.get(key)
            logger.info(f"Form[{key}]: {type(value)} - {len(str(value)) if value else 'None'} chars")
        
        # Получаем файлы из формы - пробуем разные способы
        files_data = form.getlist("files")
        if not files_data:
            # Пробуем альтернативные названия
            files_data = form.getlist("directory") 
        
        selected_extensions = form.get("selected_extensions")
        
        logger.info(f"Received upload_directory request with {len(files_data)} files")
        logger.info(f"Selected extensions parameter: {selected_extensions}")
        
        # Проверяем что файлы были загружены
        if not files_data:
            logger.error("No files received in upload_directory request")
            uploaded_files = services.get_uploaded_files()
            total_chunks = services.get_total_chunks_count()
            return templates.TemplateResponse(request, "index.html", {
                "request": request,
                "uploaded_files": uploaded_files,
                "total_chunks": total_chunks,
                "error": "No files were uploaded. Please select a directory with files."
            })
        
        # Парсим выбранные расширения из формы
        if selected_extensions and selected_extensions.strip():
            try:
                # JSON строка с выбранными расширениями
                chosen_extensions = json.loads(selected_extensions)
                logger.info(f"Parsed extensions: {chosen_extensions}")
                
                if chosen_extensions:  # Проверяем что список не пустой
                    # Добавляем точку к расширениям если её нет
                    allowed_extensions = {
                        ext if ext.startswith('.') else f'.{ext}' 
                        for ext in chosen_extensions
                    }
                    logger.info(f"Using filtered extensions: {sorted(allowed_extensions)}")
                else:
                    # Если список пустой, используем все расширения
                    allowed_extensions = get_default_extensions()
                    logger.info("Empty extensions list, using all default extensions")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Error parsing selected_extensions: {e}. Using default extensions.")
                # Если ошибка парсинга, используем все расширения по умолчанию
                allowed_extensions = get_default_extensions()
        else:
            logger.info("No extension filter provided, using all default extensions")
            # Если не выбраны расширения, используем все по умолчанию
            allowed_extensions = get_default_extensions()
        
        # Обрабатываем кастомные расширения
        custom_extensions = form.get("custom_extensions")
        if custom_extensions and custom_extensions.strip():
            try:
                # Парсим кастомные расширения (разделенные запятыми)
                custom_exts = [ext.strip() for ext in custom_extensions.split(',') if ext.strip()]
                # Добавляем точку к расширениям если её нет
                custom_exts_with_dots = {
                    ext if ext.startswith('.') else f'.{ext}' 
                    for ext in custom_exts
                }
                # Объединяем с уже выбранными расширениями
                allowed_extensions.update(custom_exts_with_dots)
                logger.info(f"Added custom extensions: {sorted(custom_exts_with_dots)}")
                logger.info(f"Final allowed extensions: {sorted(allowed_extensions)}")
            except Exception as e:
                logger.warning(f"Error parsing custom_extensions: {e}. Skipping custom extensions.")
        
        processed_files = 0
        skipped_files = 0
        empty_files = 0
        
        for file in files_data:
            # file.filename contains the relative path from the uploaded directory
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in allowed_extensions:
                skipped_files += 1
                logger.debug(f"Skipping file {file.filename} - extension {ext} not in allowed list")
                continue

            content = await file.read()
            if not content:
                empty_files += 1
                logger.debug(f"Skipping empty file: {file.filename}")
                continue  # Skip empty files

            # Sanitize the filename to be a flat path, replacing directory separators
            sanitized_filename = file.filename.replace('/', '_').replace('\\', '_')
            
            file_path = os.path.join(services.UPLOADS_DIR, sanitized_filename)
            
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            processed_files += 1
            logger.debug(f"Processed file: {file.filename}")
        
        # Логирование результатов
        logger.info(f"Directory processing summary: {processed_files} files processed, "
                    f"{skipped_files} files skipped (wrong extension), "
                    f"{empty_files} empty files skipped")
        
        uploaded_files = services.get_uploaded_files()
        total_chunks = services.get_total_chunks_count()
        return templates.TemplateResponse(request, "index.html", {
            "request": request,
            "uploaded_files": uploaded_files,
            "processed_chunks": None,
            "total_chunks": total_chunks
        })
    
    except Exception as e:
        logger.error(f"Error in upload_directory: {e}", exc_info=True)
        uploaded_files = services.get_uploaded_files()
        total_chunks = services.get_total_chunks_count()
        return templates.TemplateResponse(request, "index.html", {
            "request": request,
            "uploaded_files": uploaded_files,
            "total_chunks": total_chunks,
            "error": f"Error uploading directory: {str(e)}"
        })

@app.post("/process-files/", response_class=HTMLResponse)
async def process_files(request: Request, use_tfidf: bool = Form(True)):
    """Processes uploaded files, generates embeddings, and returns results."""
    # The checkbox sends "True" if checked, but isn't sent if not.
    # FastAPI handles bool conversion. Defaulting to True if not provided.
    processed_chunks = services.process_and_embed_files(use_tfidf_keywords=use_tfidf)
    
    # Convert to pretty-printed JSON string for display
    processed_chunks_json_str = json.dumps(processed_chunks, indent=2)
    
    # After processing, the uploads folder should be empty
    uploaded_files = services.get_uploaded_files()
    total_chunks = services.get_total_chunks_count()

    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "uploaded_files": uploaded_files,
        "processed_chunks": processed_chunks_json_str,
        "total_chunks": total_chunks
    })

@app.post("/search/", response_class=HTMLResponse)
async def search(
    request: Request, 
    query: str = Form(...),
    search_type: str = Form("hybrid"),
    use_reranker: bool = Form(True),
    expand_query: bool = Form(False)
):
    """Performs a semantic search and displays results."""
    search_results = services.search_chunks(
        query=query,
        search_type=search_type,
        use_reranker=use_reranker,
        expand_query=expand_query
    )
    uploaded_files = services.get_uploaded_files()
    total_chunks = services.get_total_chunks_count()

    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "uploaded_files": uploaded_files,
        "processed_chunks": None,
        "query": query,
        "search_results": search_results,
        "search_type": search_type,
        "use_reranker": use_reranker,
        "expand_query": expand_query,
        "total_chunks": total_chunks
    })

# Добавляем JSON API endpoint для MCP интеграции
@app.post("/api/search")
async def api_search(
    query: str,
    top_k: int = Query(5, ge=1, le=50),
    search_type: str = Query("hybrid", enum=["hybrid", "semantic", "keyword"]),
    use_reranker: bool = Query(True),
    expand_query: bool = Query(False),
    project: Optional[str] = Query(None),
    report_type: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    tags: Optional[str] = Query(None, description="Comma-separated; any-of match"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None)
):
    """
    JSON API для продвинутого поиска (для MCP интеграции)
    """
    if not query or not query.strip():
        return {"error": "Query cannot be empty", "results": []}

    try:
        # Build metadata filters (OKF: type / project / report_type / tags / date range)
        filters = {k: v for k, v in {
            "project": project, "report_type": report_type, "type": type, "tags": tags,
            "date_from": date_from, "date_to": date_to,
        }.items() if v}

        search_results = services.search_chunks(
            query=query,
            top_k=top_k,
            search_type=search_type,
            use_reranker=use_reranker,
            expand_query=expand_query,
            filters=filters or None
        )

        return {
            "query": query,
            "search_params": {
                "search_type": search_type,
                "use_reranker": use_reranker,
                "expand_query": expand_query,
                "filters": filters,
            },
            "total_results": len(search_results),
            "results": search_results
        }
        
    except Exception as e:
        return {"error": str(e), "results": []}

@app.post("/api/ingest-report")
async def api_ingest_report(
    path: str,
    project: Optional[str] = Query(None, description="Optional; inferred from path .../projects/<project>/..."),
    report_type: Optional[str] = Query(None),
    report_date: Optional[str] = Query(None),
    title: Optional[str] = Query(None),
    description: Optional[str] = Query(None),
    tags: Optional[str] = Query(None, description="Comma-separated OKF tags"),
    type: Optional[str] = Query(None, description="OKF type (default 'Report')"),
):
    """Ingest one report file (by a server-accessible path) as an OKF concept document."""
    try:
        _tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        result = services.ingest_report(path, project=project, report_type=report_type,
                                        report_date=report_date, title=title,
                                        description=description, tags=_tags, doc_type=type)
        return {"status": "success", **result}
    except FileNotFoundError:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": f"File not found: {path}"})
    except Exception as e:
        logger.error(f"ingest-report failed for {path}: {e}", exc_info=True)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/api/delete-report")
async def api_delete_report(source_path: str):
    """Delete all chunks for a report by its source_path (the path used at ingest)."""
    try:
        deleted = services.delete_report(source_path)
        return {"status": "success", "source_path": source_path, "deleted_chunks": deleted}
    except Exception as e:
        logger.error(f"delete-report failed for {source_path}: {e}", exc_info=True)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/reports")
async def api_reports():
    """List all reports with their OKF metadata (drives the browser)."""
    try:
        return {"reports": services.list_reports()}
    except Exception as e:
        return {"error": str(e), "reports": []}


@app.get("/reports", response_class=HTMLResponse)
async def reports_browser(
    request: Request,
    project: Optional[str] = Query(None),
    report_type: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
):
    """Browse all reports, grouped by project -> report_type, with simple facet filters."""
    reports = services.list_reports()
    projects = sorted({r["project"] for r in reports if r["project"]})
    types = sorted({r["report_type"] for r in reports if r["report_type"]})
    all_tags = sorted({t for r in reports for t in (r["tags"] or [])})
    if project:
        reports = [r for r in reports if r["project"] == project]
    if report_type:
        reports = [r for r in reports if r["report_type"] == report_type]
    if tag:
        reports = [r for r in reports if tag in (r["tags"] or [])]
    if q:
        ql = q.lower()
        reports = [r for r in reports
                   if ql in (r.get("title") or "").lower()
                   or ql in (r.get("description") or "").lower()]
    grouped = {}
    for r in reports:
        grouped.setdefault(r["project"] or "—", {}).setdefault(r["report_type"] or "—", []).append(r)
    return templates.TemplateResponse(request, "reports.html", {
        "grouped": grouped, "total": len(reports),
        "projects": projects, "types": types, "all_tags": all_tags,
        "f": {"project": project or "", "report_type": report_type or "", "tag": tag or "", "q": q or ""},
    })


@app.get("/reports/view", response_class=HTMLResponse)
async def report_view(request: Request, source_path: str):
    """Render one report's markdown body (read from the mounted /reports volume, read-only)."""
    import os, re
    real = os.path.realpath(source_path)
    if not (real == "/reports" or real.startswith("/reports/")) or not os.path.isfile(real):
        return HTMLResponse("Report not found", status_code=404)
    raw = open(real, encoding="utf-8", errors="replace").read()
    body = raw
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, re.DOTALL)
    meta_yaml = m.group(1) if m else ""
    if m:
        body = m.group(2)
    try:
        import markdown as _md
        content_html = _md.markdown(body, extensions=["fenced_code", "tables", "sane_lists", "nl2br"])
    except Exception:
        content_html = "<pre>" + body.replace("<", "&lt;") + "</pre>"
    return templates.TemplateResponse(request, "report_view.html", {
        "content_html": content_html, "meta_yaml": meta_yaml,
        "source_path": source_path, "name": os.path.basename(real),
    })

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """
    JSON API to upload and immediately process a single file.
    This is an atomic operation.
    """
    if not file.filename:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": "Filename is missing."})

    # Use a temporary file to leverage existing service logic
    sanitized_filename = os.path.basename(file.filename)
    # Add a unique prefix to avoid collisions during async execution
    temp_file_path = os.path.join(services.UPLOADS_DIR, f"temp_api_{sanitized_filename}")

    try:
        content = await file.read()
        if not content:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=400, content={"error": "File is empty."})

        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        # Process the single file. TF-IDF is not very useful for a single file.
        processed_chunks = services.process_single_file(temp_file_path, use_tfidf_keywords=False)
        
        return {
            "status": "success",
            "message": f"File '{sanitized_filename}' processed successfully.",
            "filename": sanitized_filename,
            "total_chunks_added": len(processed_chunks),
            "processed_chunks_preview": processed_chunks[:3] # Preview of first 3 chunks
        }
    except Exception as e:
        logger.error(f"Error processing file via API '{file.filename}': {e}", exc_info=True)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/delete-file/", response_class=HTMLResponse)
async def delete_file(request: Request, filename: str = Form(...)):
    """Удаляет загруженный файл."""
    try:
        file_path = os.path.join(services.UPLOADS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            # Возвращаем обновленный список файлов
            uploaded_files = services.get_uploaded_files()
            total_chunks = services.get_total_chunks_count()
            return templates.TemplateResponse(request, "index.html", {
                "request": request,
                "uploaded_files": uploaded_files,
                "processed_chunks": None,
                "total_chunks": total_chunks,
                "message": f"File {filename} successfully deleted"
            })
        else:
            uploaded_files = services.get_uploaded_files()
            total_chunks = services.get_total_chunks_count()
            return templates.TemplateResponse(request, "index.html", {
                "request": request,
                "uploaded_files": uploaded_files,
                "processed_chunks": None,
                "total_chunks": total_chunks,
                "error": f"File {filename} not found"
            })
    except Exception as e:
        uploaded_files = services.get_uploaded_files()
        total_chunks = services.get_total_chunks_count()
        return templates.TemplateResponse(request, "index.html", {
            "request": request,
            "uploaded_files": uploaded_files,
            "processed_chunks": None,
            "total_chunks": total_chunks,
            "error": f"Error deleting file: {str(e)}"
        })

@app.get("/api/stats")
async def api_stats():
    """
    JSON API для получения статистики (для MCP интеграции)
    """
    try:
        total_chunks = services.get_total_chunks_count()
        uploaded_files = services.get_uploaded_files()
        
        return {
            "total_chunks": total_chunks,
            "uploaded_files": len(uploaded_files),
            "files_pending": uploaded_files
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers
    """
    try:
        # Check if services are initialized
        if services.model is None or services.db_connection is None:
            return {"status": "initializing", "message": "Services are being initialized"}
        
        # Check database connection
        total_chunks = services.get_total_chunks_count()
        
        return {
            "status": "healthy",
            "message": "All services are operational",
            "total_chunks": total_chunks,
            "model_loaded": services.model is not None,
            "db_connected": services.db_connection is not None
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Service error: {str(e)}",
            "error": str(e)
        } 