"""
Tests for RAG system content-query handling
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem


class TestRAGSystemContentQueryHandling:
    """Test cases for RAG system content-query handling"""

    @pytest.fixture
    def mock_rag_system(self, test_config):
        """Create RAG system with mocked components"""
        with (
            patch("rag_system.VectorStore") as mock_vs,
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.DocumentProcessor") as mock_doc,
            patch("rag_system.SessionManager") as mock_session,
        ):

            rag = RAGSystem(test_config)

            # Configure mocks
            rag.vector_store = mock_vs.return_value
            rag.ai_generator = mock_ai.return_value
            rag.document_processor = mock_doc.return_value
            rag.session_manager = mock_session.return_value

            # Mock tool manager methods properly
            rag.tool_manager.get_last_sources = Mock(return_value=[])
            rag.tool_manager.reset_sources = Mock()
            rag.tool_manager.get_tool_definitions = Mock(return_value=[])

            # Mock search tool behavior
            rag.search_tool.last_sources = []

            return rag

    def test_query_basic_content_search(self, mock_rag_system):
        """Test basic content search query processing"""
        # Mock AI response indicating tool use
        mock_rag_system.ai_generator.generate_response.return_value = (
            "Machine learning is a powerful technology..."
        )

        # Mock tool manager sources
        mock_sources = [
            {"text": "ML Course - Lesson 1", "url": "https://example.com/lesson1"}
        ]
        mock_rag_system.tool_manager.get_last_sources.return_value = mock_sources

        response, sources = mock_rag_system.query("What is machine learning?")

        assert response == "Machine learning is a powerful technology..."
        assert sources == mock_sources

        # Verify AI generator was called with tools
        mock_rag_system.ai_generator.generate_response.assert_called_once()
        call_args = mock_rag_system.ai_generator.generate_response.call_args

        assert "Answer this question about course materials:" in call_args[1]["query"]
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None

    def test_query_with_session_management(self, mock_rag_system):
        """Test query processing with session management"""
        session_id = "test_session_123"

        # Mock conversation history
        mock_history = "User: Hello\nAssistant: Hi there!"
        mock_rag_system.session_manager.get_conversation_history.return_value = (
            mock_history
        )

        mock_rag_system.ai_generator.generate_response.return_value = (
            "Follow-up response"
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = mock_rag_system.query(
            "Follow-up question", session_id=session_id
        )

        assert response == "Follow-up response"

        # Verify session history was retrieved
        mock_rag_system.session_manager.get_conversation_history.assert_called_with(
            session_id
        )

        # Verify conversation was updated
        mock_rag_system.session_manager.add_exchange.assert_called_with(
            session_id, "Follow-up question", "Follow-up response"
        )

        # Verify history was passed to AI
        call_args = mock_rag_system.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == mock_history

    def test_query_without_session(self, mock_rag_system):
        """Test query processing without session management"""
        mock_rag_system.ai_generator.generate_response.return_value = (
            "Response without session"
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = mock_rag_system.query("Test question")

        assert response == "Response without session"

        # Verify no session methods were called
        mock_rag_system.session_manager.get_conversation_history.assert_not_called()
        mock_rag_system.session_manager.add_exchange.assert_not_called()

        # Verify no history was passed to AI
        call_args = mock_rag_system.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] is None

    def test_query_sources_reset_after_retrieval(self, mock_rag_system):
        """Test that sources are reset after being retrieved"""
        mock_sources = [{"text": "Test source", "url": "https://example.com"}]
        mock_rag_system.tool_manager.get_last_sources.return_value = mock_sources
        mock_rag_system.ai_generator.generate_response.return_value = "Test response"

        response, sources = mock_rag_system.query("Test question")

        # Verify sources were retrieved and reset
        mock_rag_system.tool_manager.get_last_sources.assert_called_once()
        mock_rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_prompt_formatting(self, mock_rag_system):
        """Test that query prompt is properly formatted"""
        test_query = "What is deep learning?"
        mock_rag_system.ai_generator.generate_response.return_value = "Response"
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        mock_rag_system.query(test_query)

        call_args = mock_rag_system.ai_generator.generate_response.call_args
        prompt = call_args[1]["query"]

        assert "Answer this question about course materials:" in prompt
        assert test_query in prompt

    def test_add_course_document_success(
        self, mock_rag_system, sample_course, sample_course_chunks
    ):
        """Test successful addition of course document"""
        # Mock document processor
        mock_rag_system.document_processor.process_course_document.return_value = (
            sample_course,
            sample_course_chunks,
        )

        course, chunk_count = mock_rag_system.add_course_document("/path/to/course.txt")

        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)

        # Verify vector store operations
        mock_rag_system.vector_store.add_course_metadata.assert_called_once_with(
            sample_course
        )
        mock_rag_system.vector_store.add_course_content.assert_called_once_with(
            sample_course_chunks
        )

    def test_add_course_document_processing_error(self, mock_rag_system):
        """Test handling of document processing errors"""
        # Mock processing error
        mock_rag_system.document_processor.process_course_document.side_effect = (
            Exception("Processing failed")
        )

        course, chunk_count = mock_rag_system.add_course_document("/invalid/path.txt")

        assert course is None
        assert chunk_count == 0

        # Verify vector store operations were not called
        mock_rag_system.vector_store.add_course_metadata.assert_not_called()
        mock_rag_system.vector_store.add_course_content.assert_not_called()

    def test_add_course_folder_with_clear_existing(self, mock_rag_system, temp_dir):
        """Test adding course folder with clear_existing=True"""
        # Create test files
        course1_path = os.path.join(temp_dir, "course1.txt")
        course2_path = os.path.join(temp_dir, "course2.txt")

        with open(course1_path, "w") as f:
            f.write("Test course 1 content")
        with open(course2_path, "w") as f:
            f.write("Test course 2 content")

        # Mock existing courses
        mock_rag_system.vector_store.get_existing_course_titles.return_value = []

        # Mock document processing
        def mock_process(file_path):
            course_name = os.path.basename(file_path).replace(".txt", "")
            course = Course(title=f"Test Course {course_name}")
            chunks = [
                CourseChunk(content="test", course_title=course.title, chunk_index=0)
            ]
            return course, chunks

        mock_rag_system.document_processor.process_course_document.side_effect = (
            mock_process
        )

        total_courses, total_chunks = mock_rag_system.add_course_folder(
            temp_dir, clear_existing=True
        )

        assert total_courses == 2
        assert total_chunks == 2

        # Verify data was cleared
        mock_rag_system.vector_store.clear_all_data.assert_called_once()

    def test_add_course_folder_skip_existing(self, mock_rag_system, temp_dir):
        """Test that existing courses are skipped"""
        # Create test file
        course_path = os.path.join(temp_dir, "existing_course.txt")
        with open(course_path, "w") as f:
            f.write("Test content")

        # Mock existing course
        existing_course = Course(title="Existing Course")
        mock_rag_system.vector_store.get_existing_course_titles.return_value = [
            "Existing Course"
        ]
        mock_rag_system.document_processor.process_course_document.return_value = (
            existing_course,
            [],
        )

        total_courses, total_chunks = mock_rag_system.add_course_folder(temp_dir)

        assert total_courses == 0
        assert total_chunks == 0

        # Verify course was not added to vector store
        mock_rag_system.vector_store.add_course_metadata.assert_not_called()

    def test_add_course_folder_nonexistent_path(self, mock_rag_system):
        """Test handling of nonexistent folder path"""
        total_courses, total_chunks = mock_rag_system.add_course_folder(
            "/nonexistent/path"
        )

        assert total_courses == 0
        assert total_chunks == 0

    def test_get_course_analytics(self, mock_rag_system):
        """Test course analytics retrieval"""
        mock_rag_system.vector_store.get_course_count.return_value = 5
        mock_rag_system.vector_store.get_existing_course_titles.return_value = [
            "Course 1",
            "Course 2",
            "Course 3",
        ]

        analytics = mock_rag_system.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 3
        assert "Course 1" in analytics["course_titles"]

    def test_end_to_end_content_query_workflow(self, mock_rag_system):
        """Test complete end-to-end workflow for content queries"""
        session_id = "e2e_session"

        # Mock session history
        mock_rag_system.session_manager.get_conversation_history.return_value = None

        # Mock AI response with tool usage
        mock_rag_system.ai_generator.generate_response.return_value = "Machine learning is a subset of AI that enables computers to learn from data."

        # Mock sources from tool execution
        mock_sources = [
            {
                "text": "ML Fundamentals - Lesson 1",
                "url": "https://example.com/ml-lesson1",
            },
            {"text": "AI Basics - Lesson 3", "url": "https://example.com/ai-lesson3"},
        ]
        mock_rag_system.tool_manager.get_last_sources.return_value = mock_sources

        # Execute query
        response, sources = mock_rag_system.query(
            "What is machine learning and how does it work?", session_id=session_id
        )

        # Verify complete workflow
        assert (
            response
            == "Machine learning is a subset of AI that enables computers to learn from data."
        )
        assert sources == mock_sources

        # Verify AI was called with proper tools and prompt
        ai_call = mock_rag_system.ai_generator.generate_response.call_args
        assert "Answer this question about course materials:" in ai_call[1]["query"]
        assert (
            ai_call[1]["tools"] == mock_rag_system.tool_manager.get_tool_definitions()
        )
        assert ai_call[1]["tool_manager"] == mock_rag_system.tool_manager

        # Verify session was updated
        mock_rag_system.session_manager.add_exchange.assert_called_with(
            session_id,
            "What is machine learning and how does it work?",
            "Machine learning is a subset of AI that enables computers to learn from data.",
        )

        # Verify sources were reset
        mock_rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_error_handling_in_ai_generation(self, mock_rag_system):
        """Test error handling when AI generation fails"""
        # Mock AI generator to raise exception
        mock_rag_system.ai_generator.generate_response.side_effect = Exception(
            "API Error"
        )

        # Should not crash, but may return None or empty response
        with pytest.raises(Exception):
            mock_rag_system.query("Test question")

    def test_tool_manager_initialization(self, test_config):
        """Test that RAG system properly initializes tool manager with tools"""
        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.SessionManager"),
        ):

            rag = RAGSystem(test_config)

            # Verify tools are registered
            tool_definitions = rag.tool_manager.get_tool_definitions()
            tool_names = [tool["name"] for tool in tool_definitions]

            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names
            assert len(tool_definitions) == 2

    def test_query_different_question_types(self, mock_rag_system):
        """Test handling of different types of questions"""
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        # Content-specific question
        mock_rag_system.ai_generator.generate_response.return_value = "Content answer"
        response, _ = mock_rag_system.query("Explain neural networks")
        assert response == "Content answer"

        # Course outline question
        mock_rag_system.ai_generator.generate_response.return_value = "Outline answer"
        response, _ = mock_rag_system.query("What are the lessons in the ML course?")
        assert response == "Outline answer"

        # General question
        mock_rag_system.ai_generator.generate_response.return_value = "General answer"
        response, _ = mock_rag_system.query("What is the weather like?")
        assert response == "General answer"

        # Verify all queries were processed with tools available
        assert mock_rag_system.ai_generator.generate_response.call_count == 3

    def test_concurrent_session_handling(self, mock_rag_system):
        """Test handling of multiple concurrent sessions"""
        session1 = "session_1"
        session2 = "session_2"

        # Mock different histories for different sessions
        def mock_get_history(session_id):
            if session_id == session1:
                return "Session 1 history"
            elif session_id == session2:
                return "Session 2 history"
            return None

        mock_rag_system.session_manager.get_conversation_history.side_effect = (
            mock_get_history
        )
        mock_rag_system.ai_generator.generate_response.return_value = "Response"
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        # Query from both sessions
        mock_rag_system.query("Question 1", session_id=session1)
        mock_rag_system.query("Question 2", session_id=session2)

        # Verify correct histories were used
        history_calls = (
            mock_rag_system.session_manager.get_conversation_history.call_args_list
        )
        assert len(history_calls) == 2
        assert history_calls[0][0][0] == session1
        assert history_calls[1][0][0] == session2

        # Verify both sessions were updated
        exchange_calls = mock_rag_system.session_manager.add_exchange.call_args_list
        assert len(exchange_calls) == 2
