# Engram — Reports Knowledge Hub (RAG + RLM)

> Formerly `RAG-DuckDB-with-MCP`. **Engram** is a local, sovereign hub that centralizes project reports outside their repos and makes them hybrid-searchable (RAG), with a recursive-reasoning (RLM) layer planned on top.

Engram is a Python server for document processing and retrieval-augmented generation (RAG). It provides a web interface and a JSON API to upload documents, chunk them, generate embeddings, and store them in DuckDB (VSS + FTS) for hybrid similarity search.

The application is containerized with Docker and uses `uv` for fast, optimized dependency management. It also includes an `mcp-rag-service` for integration with MCP (Model Context Protocol).

## Features

-   **Web Interface**: Minimalist UI for uploading files, initiating processing, and performing searches.
-   **JSON API**: Provides `/api/search`, `/api/stats`, and `/health` endpoints for programmatic integration.
-   **Wide File Support**: Handles various file types including `.txt`, `.md`, `.pdf`, and multiple programming language source files (`.py`, `.js`, `.java`, etc.).
-   **Advanced Chunking**: Uses different strategies based on file type (e.g., `CodeSplitter` for source code, `RecursiveCharacterTextSplitter` for text).
-   **High-Quality Embeddings**: Uses `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (primary, 768d) or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (fallback, 384d).
-   **Vector Database**: Leverages DuckDB with the VSS (Vector Similarity Search) extension for efficient storage and querying of embeddings.
-   **Dockerized & Optimized**:
    -   Easy to build and run with Docker.
    -   Uses `uv` for ultra-fast dependency installation.
    -   Multi-stage Dockerfile for small final image size.
    -   Supports CPU-only builds for environments without a GPU.
-   **MCP Integration**: Includes a sample `mcp-rag-service` to demonstrate integration with external systems.
-   **Directory Upload**: Support for uploading entire directories with file extension filtering.
-   **Health Monitoring**: Built-in health check endpoint for monitoring and load balancers.

## Tech Stack

-   **Backend**: Python with FastAPI
-   **Embeddings**: `sentence-transformers`, `llama-index`, `langchain`
-   **Database**: DuckDB + VSS extension
-   **Containerization**: Docker
-   **Package Management**: `uv`

## How to Run

### Prerequisites

-   Docker installed and running on your machine.

### Build and Run the Docker Container

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Build the Docker image:**
    The build process is optimized using a multi-stage Dockerfile and `uv`. You can choose between a standard build (which includes GPU-capable libraries) and a CPU-only build.

    **Standard Build (for environments with GPU support):**
    ```bash
    docker build -t rag-duckdb-server .
    ```

    **CPU-Only Build (recommended for local development or CPU servers):**
    This build is faster and results in a smaller image by using a CPU-only version of PyTorch.
    ```bash
    docker build --build-arg USE_CPU_ONLY=true -t rag-duckdb-server-cpu .
    ```

3.  **Run the Docker container:**
    This command starts the server and maps the local `uploads` and `data` directories to the container. This ensures your uploaded files and the database persist even if the container is removed.

    *For standard build:*
    ```bash
    docker run -p 8000:8000 \
      -v "$(pwd)/uploads:/app/uploads" \
      -v "$(pwd)/data:/app/data" \
      --name rag-server \
      rag-duckdb-server
    ```
    *For CPU-only build:*
    ```bash
    docker run -p 8000:8000 \
      -v "$(pwd)/uploads:/app/uploads" \
      -v "$(pwd)/data:/app/data" \
      --name rag-server-cpu \
      rag-duckdb-server-cpu
    ```
    *Note for Windows users*: Use `${pwd}` instead of `$(pwd)` in PowerShell.

4.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8000`.

## Usage Workflow

1.  **Upload Files**: Use the web interface to select and upload one or more supported files.
2.  **Upload Directory**: Alternatively, upload entire directories with file extension filtering to process only specific file types.
3.  **Process Files**: Click the "Start Processing" button. The server will:
    -   Extract text content.
    -   Split the text into manageable, context-aware chunks.
    -   Generate a vector embedding for each chunk.
    -   Save the chunks and their embeddings to the `data/rag.duckdb` database.
    -   Delete processed files from the `uploads` folder.
4.  **Search Documents**: Once documents are processed, use the semantic search bar to find relevant content across all indexed chunks.
5.  **Use API**: Interact with the server programmatically via the `/api/*` endpoints.

## Supported File Types

The server supports a wide range of file types:

### Text Documents
- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents

### Programming Languages
- `.py` - Python
- `.js`, `.ts`, `.jsx`, `.tsx` - JavaScript/TypeScript
- `.java` - Java
- `.c`, `.cpp`, `.cc`, `.cxx` - C/C++
- `.cs` - C#
- `.go` - Go
- `.rs` - Rust
- `.php` - PHP
- `.rb` - Ruby
- `.scala` - Scala
- `.swift` - Swift

### Web Technologies
- `.html`, `.htm` - HTML
- `.css`, `.scss`, `.sass` - CSS and preprocessors

### Shell Scripts
- `.sh`, `.bash`, `.zsh`, `.fish` - Shell scripts

### Data Formats
- `.json` - JSON
- `.yaml`, `.yml` - YAML
- `.xml` - XML
- `.sql` - SQL
- `.ini`, `.toml` - Configuration files

**Note**: Files with unsupported extensions are automatically skipped during processing.

## API Endpoints

### Web Interface
- `GET /` - Main web interface
- `POST /upload-files/` - Upload individual files
- `POST /upload-directory/` - Upload directory with extension filtering
- `POST /process-files/` - Process uploaded files
- `POST /search/` - Search interface
- `POST /delete-file/` - Delete uploaded file

### JSON API
- `POST /api/search` - Programmatic search endpoint
- `GET /api/stats` - Get collection statistics
- `GET /health` - Health check endpoint

### Search API Parameters
- `query` (required): Search query string
- `top_k` (optional, default: 5): Number of results to return (1-50)
- `search_type` (optional, default: "hybrid"): "hybrid", "semantic", or "keyword"
- `use_reranker` (optional, default: true): Enable/disable result reranking
- `expand_query` (optional, default: false): Enable/disable query expansion

## MCP Integration

The project includes a separate MCP (Machine Comprehension Platform) integration service located in the `mcp-rag-service/` directory. This service provides:

- **RAG Client**: Python client for interacting with the RAG server
- **Vector Analytics**: Advanced analysis capabilities including clustering, outlier detection, and similarity matrices
- **MCP Server**: Integration with MCP-compatible tools

### MCP Examples

The `mcp-rag-service/examples/` directory contains working examples:

- `upload_example.py` - Demonstrates file upload functionality
- `search_example.py` - Shows semantic search with similarity thresholds
- `analysis_example.py` - Comprehensive vector analysis examples

To run the examples:
```bash
cd mcp-rag-service/examples
python upload_example.py
python search_example.py
python analysis_example.py
```

## Project Structure

```
.
├── app/
│   ├── main.py           # FastAPI application, routes, and API endpoints
│   └── services.py       # Business logic (file processing, chunking, embeddings, DB)
├── mcp-rag-service/      # MCP integration service
│   ├── src/
│   │   ├── rag_client.py         # RAG server client
│   │   ├── rag_mcp_server.py     # MCP server implementation
│   │   ├── vector_operations.py  # Advanced vector analytics
│   │   └── utils.py              # Utility functions
│   ├── examples/                 # Working examples
│   └── pyproject.toml
├── templates/
│   └── index.html        # Jinja2 template for the UI
├── uploads/              # Directory for file uploads (mounted as a volume)
├── data/                 # Directory for DuckDB database (mounted as a volume)
├── .dockerignore         # Specifies files to ignore in Docker build context
├── .gitignore            # Specifies files to ignore for Git
├── Dockerfile            # Docker build instructions with uv and multi-stage builds
├── requirements-base.txt # Base Python dependencies
├── requirements-cpu.txt  # CPU-only ML dependencies
├── requirements-ml.txt   # Full ML dependencies (for GPU)
└── README.md             # This file
```

## Configuration

- **Embedding Models**: The primary and fallback models are defined as constants in `app/services.py`.
- **Chunking**: Chunk size and overlap can be adjusted via the `CHUNK_SIZE` and `CHUNK_OVERLAP` environment variables. The defaults are 700 and 100, respectively.
- **Database Path**: The path to the DuckDB file is configured in `app/services.py`.
- **Search Features**: The UI allows for advanced search configuration:
    - **Search Type**: Choose between `Hybrid` (Semantic + Keyword), `Semantic`-only, or `Keyword`-only (BM25) search.
    - **Reranking**: A Cross-Encoder model can be used to rerank the top search results for higher accuracy. This can be toggled in the UI.
    - **Query Expansion**: Automatically expand your query with relevant terms found from an initial search. This can be toggled in the UI.
- **Processing Features**:
    - **TF-IDF Keywords**: When processing files, you can choose to generate and attach relevant keywords to each chunk's metadata using TF-IDF. This can improve keyword-based searches.

## Error Handling

- **Unsupported Files**: Files with unsupported extensions are automatically skipped during upload and processing.
- **Empty Files**: Empty or unreadable files are automatically removed from the uploads directory.
- **Processing Errors**: Individual file processing errors are logged but don't stop the overall process.
- **API Errors**: All API endpoints return structured error responses with appropriate HTTP status codes.

## Known Limitations

- **File Size**: Very large files may cause memory issues during processing.
- **Concurrent Users**: The current implementation is designed for single-user scenarios.
- **File Formats**: Only text-based files are supported. Binary files (images, videos, etc.) are not supported.
- **Language Support**: While the embedding model is multilingual, chunking strategies are optimized for English and common programming languages.

## Roadmap & Future Plans

### Planned Features
- **GraphRAG Integration**: Advanced graph-based retrieval and reasoning capabilities
- **Multi-user Support**: User authentication and isolated document collections
- **Real-time Processing**: WebSocket support for real-time processing updates
- **Advanced Analytics**: More sophisticated vector analysis and visualization tools
- **Plugin System**: Extensible architecture for custom processors and analyzers
- **Performance Optimization**: Caching, indexing improvements, and distributed processing

### GraphRAG Implementation
GraphRAG (Graph-based Retrieval-Augmented Generation) is planned as a major enhancement that will provide:
- **Knowledge Graph Construction**: Automatic extraction of entities and relationships
- **Graph-based Retrieval**: Enhanced search using graph traversal and reasoning
- **Multi-hop Reasoning**: Complex queries that require multiple reasoning steps
- **Contextual Understanding**: Better understanding of document relationships and hierarchies

This feature is currently in the planning phase and will be implemented as a separate module that can be optionally enabled.

## Troubleshooting

### Common Issues

1. **Docker Build Fails**: Try the CPU-only build for faster, more reliable builds:
   ```bash
   docker build --build-arg USE_CPU_ONLY=true -t rag-duckdb-server-cpu .
   ```

2. **Memory Issues**: For large document collections, consider:
   - Using the CPU-only build (smaller memory footprint)
   - Processing files in smaller batches
   - Increasing Docker memory limits

3. **Model Loading Issues**: The system automatically falls back to a smaller model if the primary model fails to load.

4. **Database Issues**: The DuckDB database is automatically created on first run. If you encounter database errors, you can delete the `data/` directory to start fresh.

### Health Check

Use the health check endpoint to monitor service status:
```bash
curl http://localhost:8000/health
```

This returns service status, model loading state, and database connection information.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
