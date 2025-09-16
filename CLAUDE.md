# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install all dependencies
uv sync

# Add new dependency
uv add package_name

# Update dependencies
uv sync --upgrade
```

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit to add ANTHROPIC_API_KEY
nano .env
```

### Development Workflow

```bash
# Add new course documents (place .txt files in docs/ directory)
# Files are automatically loaded on server startup
# Expected format: Course Title/Link/Instructor + Lessons with content

# Reset/clear vector database
rm -rf backend/chroma_db

# Verify system is working
curl http://localhost:8000/api/courses
```

### Testing
Currently no test framework is configured. Future implementations should use:
```bash
# If tests are added in the future
uv add pytest
uv run pytest
```

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) System** for querying course materials using AI-powered semantic search.

### Core Components

**RAG System (`rag_system.py`)**: Main orchestrator that coordinates all components
- Processes user queries through the complete RAG pipeline
- Manages document ingestion and vector storage
- Orchestrates AI generation with tool-based search

**Vector Store (`vector_store.py`)**: ChromaDB-based semantic search engine
- Two collections: `course_catalog` (metadata) and `course_content` (chunks)
- Uses `all-MiniLM-L6-v2` sentence transformer for embeddings
- Supports fuzzy course name resolution and metadata filtering

**Document Processor (`document_processor.py`)**: Structured content extraction
- Parses course documents with expected format: title/link/instructor + lessons
- Implements sentence-aware chunking (800 chars, 100 char overlap)
- Adds contextual prefixes: "Course X Lesson Y content: ..."

**AI Generator (`ai_generator.py`)**: Claude API integration with tools
- Uses tool calling for autonomous search decisions
- Handles multi-turn conversations with tool execution
- Temperature 0, max 800 tokens for consistent responses

**Search Tools (`search_tools.py`)**: Tool interface for Claude
- `CourseSearchTool` provides semantic search capability to AI
- Tool manager handles registration and execution
- Tracks sources for UI attribution

### Data Flow Architecture

1. **Document Ingestion**: Text files → Structured parsing → Chunking → Vector embeddings → ChromaDB storage
2. **Query Processing**: User input → Session context → AI with tools → Vector search (if needed) → Response synthesis
3. **Frontend Integration**: FastAPI serves both API endpoints and static files for the web interface

### API Endpoints

**POST `/api/query`**: Process user queries with optional session management
- Request: `{"query": "string", "session_id": "optional_string"}`
- Response: `{"answer": "string", "sources": ["string"], "session_id": "string"}`

**GET `/api/courses`**: Get course statistics and analytics
- Response: `{"total_courses": int, "course_titles": ["string"]}`

**GET `/`**: Web interface (served from `/frontend/` directory)

### Key Architectural Decisions

**Two-Collection Vector Strategy**: Separates course metadata from content for efficient filtering and search
**Tool-Based AI**: Claude autonomously decides when to search rather than always retrieving context
**Session Management**: In-memory conversation tracking with configurable history limits (default: 2 exchanges)
**Context-Enhanced Chunks**: Each chunk includes course/lesson metadata for better retrieval accuracy

### File Structure Conventions

**Backend Components**: All Python modules in `/backend/` with clear separation of concerns
**Frontend**: Vanilla HTML/CSS/JS in `/frontend/` served by FastAPI static files
**Documents**: Course materials in `/docs/` as `.txt` files with structured format
**Configuration**: Centralized in `config.py` with environment variable support

### Development Patterns

**Document Format**: Required structure for course files in `/docs/` directory:
```
Course Title: Building Towards Computer Use with Anthropic
Course Link: https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/
Course Instructor: Colt Steele

Lesson 0: Introduction
Lesson Link: https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction
Welcome to Building Toward Computer Use with Anthropic. Built in partnership with Anthropic...

Lesson 1: Next Lesson Title
Lesson Link: [lesson url]
[lesson content...]
```

**File Naming**: Use descriptive names like `course1_script.txt`, `python_fundamentals.txt`

**Error Handling**: Each component returns structured errors rather than exceptions for graceful degradation

**API Design**: RESTful endpoints (`/api/query`, `/api/courses`) with Pydantic models for request/response validation

## Troubleshooting

### Common Issues

**"ANTHROPIC_API_KEY not found"**: Ensure `.env` file exists with valid API key
```bash
# Check if .env exists
ls -la .env
# Should contain: ANTHROPIC_API_KEY=your_key_here
```

**ChromaDB Permission Errors**: Reset the database directory
```bash
rm -rf backend/chroma_db
# Restart server to recreate
```

**No courses loaded**: Verify documents are in correct format and location
```bash
# Check docs directory
ls docs/
# Files should be .txt with proper course format (see Development Patterns below)
```

**Port 8000 already in use**: Find and kill existing process
```bash
lsof -ti:8000 | xargs kill -9
```

### Development Notes

When working with this codebase, focus on maintaining the modular architecture where each component has a single responsibility in the RAG pipeline.

**Critical Dependencies**:
- Always use `uv` for package management (not pip directly)
- Use `uv run` prefix for Python commands
- ChromaDB storage auto-created in `backend/chroma_db/`
- Server runs on port 8000 by default