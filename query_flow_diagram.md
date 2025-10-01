# RAG System Query Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend (JS)
    participant API as FastAPI Backend
    participant RAG as RAG System
    participant Session as Session Manager
    participant AI as AI Generator (Claude)
    participant Tools as Tool Manager
    participant Search as Course Search Tool
    participant Vector as Vector Store
    participant ChromaDB as ChromaDB

    %% User initiates query
    User->>Frontend: Types query & clicks Send
    Frontend->>Frontend: Disable input, show loading
    Frontend->>Frontend: Add user message to chat

    %% HTTP request to backend
    Frontend->>+API: POST /api/query<br/>{query, session_id}

    %% Backend processing
    API->>+RAG: query(query, session_id)

    %% Session management
    RAG->>+Session: get_conversation_history(session_id)
    Session-->>-RAG: Previous messages formatted

    %% AI generation with tools
    RAG->>+AI: generate_response(query, history, tools, tool_manager)

    %% Claude API call with tools
    AI->>+AI: Call Claude API with tools available
    Note over AI: Claude decides: "I need to search for course content"
    AI->>+AI: Claude returns tool_use response

    %% Tool execution
    AI->>+Tools: execute_tool("search_course_content", params)
    Tools->>+Search: execute(query, course_name, lesson_number)

    %% Vector search process
    Search->>+Vector: search(query, course_name, lesson_number)
    Vector->>Vector: _resolve_course_name() if needed
    Vector->>Vector: _build_filter() for metadata
    Vector->>+ChromaDB: query(query_texts, n_results, where)
    ChromaDB-->>-Vector: Similar chunks with metadata
    Vector-->>-Search: SearchResults object

    %% Format and return search results
    Search->>Search: _format_results() with context
    Search->>Search: Store sources in last_sources
    Search-->>-Tools: Formatted search results
    Tools-->>-AI: Tool execution results

    %% Final AI response
    AI->>AI: Call Claude again with tool results
    AI-->>-RAG: Final generated response

    %% Collect sources and update session
    RAG->>+Tools: get_last_sources()
    Tools-->>-RAG: Source references
    RAG->>+Session: add_exchange(session_id, query, response)
    Session-->>-RAG: History updated

    %% Return to frontend
    RAG-->>-API: (response, sources)
    API-->>-Frontend: QueryResponse{answer, sources, session_id}

    %% Frontend updates
    Frontend->>Frontend: Update session_id if new
    Frontend->>Frontend: Remove loading, add AI response
    Frontend->>Frontend: Add sources as collapsible section
    Frontend->>Frontend: Re-enable input, scroll to bottom
    Frontend-->>User: Display AI response with sources

    %% Error handling paths (dotted lines)
    Note over API,ChromaDB: Error handling at each step
    Vector-->>Search: SearchResults.empty(error_msg)
    Search-->>Tools: "No relevant content found"
    Tools-->>AI: Error message
    AI-->>RAG: Response without sources
```

## Component Architecture

```mermaid
graph TB
    %% Frontend Layer
    subgraph "Frontend Layer"
        UI[HTML/CSS Interface]
        JS[JavaScript Controller]
        Chat[Chat Messages]
        Input[User Input]
    end

    %% Backend API Layer
    subgraph "Backend API Layer"
        FastAPI[FastAPI Server]
        Endpoints["/api/query, /api/courses"]
        CORS[CORS Middleware]
    end

    %% RAG System Core
    subgraph "RAG System Core"
        RAGSys[RAG System Orchestrator]
        DocProc[Document Processor]
        Config[Configuration]
    end

    %% AI & Tools Layer
    subgraph "AI & Tools Layer"
        AIGen[AI Generator]
        Claude[Anthropic Claude API]
        ToolMgr[Tool Manager]
        SearchTool[Course Search Tool]
    end

    %% Data & Session Layer
    subgraph "Data & Session Layer"
        VectorStore[Vector Store]
        ChromaDB[(ChromaDB)]
        SessionMgr[Session Manager]
        Memory[(In-Memory Sessions)]
    end

    %% Document Storage
    subgraph "Document Storage"
        Docs[/docs/*.txt files]
        Chunks[Text Chunks]
        Embeddings[Vector Embeddings]
    end

    %% Connections
    UI --> JS
    JS --> FastAPI
    FastAPI --> RAGSys
    RAGSys --> AIGen
    RAGSys --> SessionMgr
    AIGen --> Claude
    AIGen --> ToolMgr
    ToolMgr --> SearchTool
    SearchTool --> VectorStore
    VectorStore --> ChromaDB
    SessionMgr --> Memory
    DocProc --> VectorStore
    Docs --> DocProc
    DocProc --> Chunks
    Chunks --> Embeddings
    Embeddings --> ChromaDB

    %% Styling
    classDef frontend fill:#e1f5fe
    classDef backend fill:#f3e5f5
    classDef ai fill:#fff3e0
    classDef data fill:#e8f5e8
    classDef storage fill:#fce4ec

    class UI,JS,Chat,Input frontend
    class FastAPI,Endpoints,CORS,RAGSys,DocProc,Config backend
    class AIGen,Claude,ToolMgr,SearchTool ai
    class VectorStore,ChromaDB,SessionMgr,Memory data
    class Docs,Chunks,Embeddings storage
```

## Data Flow Summary

### 1. **Query Initiation**
- User types → Frontend captures → HTTP POST to `/api/query`

### 2. **Session Context**
- Session Manager retrieves conversation history
- Provides context for AI generation

### 3. **AI Processing**
- Claude receives query + history + available tools
- Decides autonomously whether to search or answer directly

### 4. **Vector Search** (if triggered)
- Course name resolution via semantic similarity
- Content search with metadata filtering
- ChromaDB returns top 5 similar chunks

### 5. **Response Generation**
- Claude synthesizes search results into coherent answer
- Sources tracked and returned separately

### 6. **Frontend Display**
- Markdown rendering for rich formatting
- Collapsible sources section
- Session continuity maintained

### Key Features:
- **Autonomous Tool Use**: Claude decides when to search
- **Semantic Matching**: Fuzzy course name resolution
- **Context Preservation**: Session-based conversation memory
- **Source Attribution**: Transparent information sourcing
- **Error Resilience**: Graceful fallbacks at each layer