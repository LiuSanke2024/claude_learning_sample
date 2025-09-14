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

**Document Format**: Expected structure for course files:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [lesson title]
Lesson Link: [lesson url]
[lesson content...]
```

**Error Handling**: Each component returns structured errors rather than exceptions for graceful degradation

**API Design**: RESTful endpoints (`/api/query`, `/api/courses`) with Pydantic models for request/response validation

When working with this codebase, focus on maintaining the modular architecture where each component has a single responsibility in the RAG pipeline.
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- use uv to run python files