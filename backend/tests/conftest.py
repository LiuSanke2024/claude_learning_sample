import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from fastapi.testclient import TestClient

# Add parent directory to path to import modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import Config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration"""
    config = Config()
    config.CHROMA_PATH = os.path.join(temp_dir, "test_chroma_db")
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.MAX_RESULTS = 3
    config.CHUNK_SIZE = 500
    config.CHUNK_OVERLAP = 50
    return config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
        Lesson(lesson_number=1, title="Basic Concepts", lesson_link="https://example.com/lesson1"),
        Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/lesson2")
    ]

    return Course(
        title="Test Course: Machine Learning Basics",
        course_link="https://example.com/course",
        instructor="Dr. Test",
        lessons=lessons
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    chunks = [
        CourseChunk(
            content="Course Test Course: Machine Learning Basics Lesson 0 content: This is an introduction to machine learning. We'll cover basic concepts and terminology.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=1
        ),
        CourseChunk(
            content="Course Test Course: Machine Learning Basics Lesson 1 content: Basic concepts include supervised and unsupervised learning methods.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=2
        ),
        CourseChunk(
            content="Course Test Course: Machine Learning Basics Lesson 2 content: Advanced topics cover neural networks and deep learning architectures.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=3
        )
    ]
    return chunks


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)

    # Mock the search method
    def mock_search(query, course_name=None, lesson_number=None, limit=None):
        # Return different results based on query and filters
        if "no results" in query.lower():
            return SearchResults(documents=[], metadata=[], distances=[])

        # Handle lesson number filtering
        if lesson_number == 0 or "introduction" in query.lower():
            return SearchResults(
                documents=["This is an introduction to machine learning. We'll cover basic concepts and terminology."],
                metadata=[{"course_title": "Test Course: Machine Learning Basics", "lesson_number": 0}],
                distances=[0.1]
            )
        elif lesson_number == 2 or "neural networks" in query.lower() or "advanced topics" in query.lower():
            return SearchResults(
                documents=["Advanced topics cover neural networks and deep learning architectures."],
                metadata=[{"course_title": "Test Course: Machine Learning Basics", "lesson_number": 2}],
                distances=[0.2]
            )
        else:
            # Default to lesson 1
            return SearchResults(
                documents=["Basic concepts include supervised and unsupervised learning methods."],
                metadata=[{"course_title": "Test Course: Machine Learning Basics", "lesson_number": 1}],
                distances=[0.3]
            )

    mock_store.search = Mock(side_effect=mock_search)

    # Mock course name resolution
    def mock_resolve_course_name(course_name):
        if "machine learning" in course_name.lower() or "test course" in course_name.lower():
            return "Test Course: Machine Learning Basics"
        return None

    mock_store._resolve_course_name = Mock(side_effect=mock_resolve_course_name)

    # Mock link methods
    mock_store.get_lesson_link = Mock(return_value="https://example.com/lesson1")
    mock_store.get_course_link = Mock(return_value="https://example.com/course")

    return mock_store




@pytest.fixture
def search_tool(mock_vector_store):
    """Create CourseSearchTool with mock vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def outline_tool(mock_vector_store):
    """Create CourseOutlineTool with mock vector store"""
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager(search_tool, outline_tool):
    """Create ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(search_tool)
    manager.register_tool(outline_tool)
    return manager


@pytest.fixture
def ai_generator_with_mock():
    """Create AIGenerator with mock client"""
    mock_client = Mock()

    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_anthropic.return_value = mock_client
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        generator.client = mock_client
        return generator


@pytest.fixture
def test_course_document_content():
    """Sample course document content for testing"""
    return """Course Title: Test Course: Machine Learning Basics
Course Link: https://example.com/course
Course Instructor: Dr. Test

Lesson 0: Introduction
Lesson Link: https://example.com/lesson0
This is an introduction to machine learning. We'll cover basic concepts and terminology. Machine learning is a subset of artificial intelligence that focuses on algorithms.

Lesson 1: Basic Concepts
Lesson Link: https://example.com/lesson1
Basic concepts include supervised and unsupervised learning methods. Supervised learning uses labeled data while unsupervised learning finds patterns in unlabeled data.

Lesson 2: Advanced Topics
Lesson Link: https://example.com/lesson2
Advanced topics cover neural networks and deep learning architectures. Neural networks are inspired by biological neurons and can learn complex patterns.
"""


@pytest.fixture
def error_vector_store():
    """Create a vector store that returns errors for testing error handling"""
    mock_store = Mock(spec=VectorStore)

    def mock_search_with_error(query, course_name=None, lesson_number=None, limit=None):
        return SearchResults.empty("Search service unavailable")

    mock_store.search = Mock(side_effect=mock_search_with_error)
    mock_store._resolve_course_name = Mock(return_value=None)

    return mock_store


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_rag = Mock(spec=RAGSystem)

    # Mock the query method
    def mock_query(query: str, session_id: str = None):
        if "error" in query.lower():
            raise Exception("Test error occurred")

        return (
            "This is a test response about machine learning concepts.",
            [
                {
                    "text": "Test Course: Machine Learning Basics - Lesson 1",
                    "url": "https://example.com/lesson1"
                }
            ]
        )

    mock_rag.query = Mock(side_effect=mock_query)

    # Mock get_course_analytics
    mock_rag.get_course_analytics = Mock(return_value={
        "total_courses": 2,
        "course_titles": [
            "Test Course: Machine Learning Basics",
            "Advanced Deep Learning"
        ]
    })

    # Mock session manager
    mock_session_manager = Mock()
    mock_session_manager.create_session = Mock(return_value="test-session-123")
    mock_session_manager.clear_session = Mock()
    mock_rag.session_manager = mock_session_manager

    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked dependencies"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Union

    # Create test app (avoiding static file mounting issue)
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")

    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Define models inline
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, SourceItem]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class SessionClearRequest(BaseModel):
        session_id: str

    class SessionClearResponse(BaseModel):
        success: bool
        message: str

    # Define endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        session_id = request.session_id
        if not session_id:
            session_id = mock_rag_system.session_manager.create_session()

        answer, sources = mock_rag_system.query(request.query, session_id)

        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        analytics = mock_rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )

    @app.post("/api/session/clear", response_model=SessionClearResponse)
    async def clear_session(request: SessionClearRequest):
        mock_rag_system.session_manager.clear_session(request.session_id)
        return SessionClearResponse(
            success=True,
            message=f"Session {request.session_id} cleared successfully"
        )

    return app


@pytest.fixture
def api_client(test_app):
    """Create a test client for API testing"""
    return TestClient(test_app)