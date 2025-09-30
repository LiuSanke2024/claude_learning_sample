"""
API endpoint tests for the RAG system.

Tests all FastAPI endpoints including:
- POST /api/query: Query processing with session management
- GET /api/courses: Course statistics and analytics
- POST /api/session/clear: Session clearing
"""

import pytest
from fastapi import status


@pytest.mark.api
class TestQueryEndpoint:
    """Test suite for /api/query endpoint"""

    def test_query_without_session_id(self, api_client, mock_rag_system):
        """
        Test query endpoint creates a new session when none provided.

        Validates:
        - 200 status code
        - Response contains answer, sources, and session_id
        - Session manager creates new session
        - RAG system processes query
        """
        response = api_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify response content
        assert data["answer"] == "This is a test response about machine learning concepts."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test Course: Machine Learning Basics - Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/lesson1"

        # Verify session was created
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_existing_session_id(self, api_client, mock_rag_system):
        """
        Test query endpoint uses provided session ID.

        Validates:
        - 200 status code
        - Same session ID returned
        - Session manager not called to create new session
        - Query processed with existing session
        """
        session_id = "existing-session-456"
        response = api_client.post(
            "/api/query",
            json={
                "query": "Explain neural networks",
                "session_id": session_id
            }
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert data["session_id"] == session_id

        # Verify session manager was not called to create new session
        mock_rag_system.session_manager.create_session.assert_not_called()

        # Verify RAG system was called with correct parameters
        mock_rag_system.query.assert_called_once()
        call_args = mock_rag_system.query.call_args
        assert call_args[0][0] == "Explain neural networks"
        assert call_args[0][1] == session_id

    def test_query_with_error_handling(self, api_client, mock_rag_system):
        """
        Test query endpoint handles errors gracefully.

        Validates:
        - 500 status code when RAG system raises exception
        - Error detail included in response
        """
        response = api_client.post(
            "/api/query",
            json={"query": "trigger error in system"}
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        assert "Test error occurred" in data["detail"]

    def test_query_with_missing_query_field(self, api_client):
        """
        Test query endpoint validates required fields.

        Validates:
        - 422 status code for validation error
        - Error indicates missing required field
        """
        response = api_client.post(
            "/api/query",
            json={}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_query_with_empty_query_string(self, api_client, mock_rag_system):
        """
        Test query endpoint handles empty query strings.

        Validates:
        - 200 status code (backend processes empty queries)
        - Response structure maintained
        """
        response = api_client.post(
            "/api/query",
            json={"query": ""}
        )

        # Empty query should still be processed
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data


@pytest.mark.api
class TestCoursesEndpoint:
    """Test suite for /api/courses endpoint"""

    def test_get_course_stats(self, api_client, mock_rag_system):
        """
        Test courses endpoint returns analytics.

        Validates:
        - 200 status code
        - Response contains total_courses and course_titles
        - Correct course count and titles
        """
        response = api_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify response content
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Test Course: Machine Learning Basics" in data["course_titles"]
        assert "Advanced Deep Learning" in data["course_titles"]

        # Verify RAG system method was called
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_course_stats_with_no_courses(self, api_client, mock_rag_system):
        """
        Test courses endpoint when no courses are loaded.

        Validates:
        - 200 status code
        - Empty course list and zero count
        """
        # Override mock to return empty data
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = api_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []


@pytest.mark.api
class TestSessionClearEndpoint:
    """Test suite for /api/session/clear endpoint"""

    def test_clear_session_success(self, api_client, mock_rag_system):
        """
        Test session clearing endpoint.

        Validates:
        - 200 status code
        - Success response with message
        - Session manager called with correct session ID
        """
        session_id = "session-to-clear-789"
        response = api_client.post(
            "/api/session/clear",
            json={"session_id": session_id}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert "success" in data
        assert "message" in data

        # Verify response content
        assert data["success"] is True
        assert session_id in data["message"]

        # Verify session manager was called
        mock_rag_system.session_manager.clear_session.assert_called_once_with(session_id)

    def test_clear_session_missing_session_id(self, api_client):
        """
        Test session clearing requires session_id.

        Validates:
        - 422 status code for validation error
        - Error indicates missing required field
        """
        response = api_client.post(
            "/api/session/clear",
            json={}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data


@pytest.mark.api
class TestCORSConfiguration:
    """Test suite for CORS configuration"""

    def test_cors_headers_present(self, api_client):
        """
        Test CORS headers are properly configured.

        Validates:
        - CORS headers present in response
        - Proper allow origins, methods, headers configuration
        """
        response = api_client.get("/api/courses")

        # Verify CORS headers are present
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"


@pytest.mark.api
class TestRequestResponseModels:
    """Test suite for Pydantic model validation"""

    def test_query_request_validation(self, api_client):
        """
        Test QueryRequest model validates input.

        Validates:
        - Invalid data types rejected
        - Validation errors provide details
        """
        # Test with invalid query type (number instead of string)
        response = api_client.post(
            "/api/query",
            json={"query": 123}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_session_clear_request_validation(self, api_client):
        """
        Test SessionClearRequest model validates input.

        Validates:
        - Invalid data types rejected
        - Validation errors provide details
        """
        # Test with invalid session_id type (number instead of string)
        response = api_client.post(
            "/api/session/clear",
            json={"session_id": 123}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.api
class TestEndToEndFlow:
    """Test suite for complete API workflows"""

    def test_complete_conversation_flow(self, api_client, mock_rag_system):
        """
        Test complete conversation flow with session management.

        Validates:
        - Create session on first query
        - Use same session for subsequent queries
        - Clear session successfully
        """
        # First query - creates new session
        response1 = api_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )
        assert response1.status_code == status.HTTP_200_OK
        session_id = response1.json()["session_id"]

        # Second query - uses existing session
        mock_rag_system.session_manager.create_session.reset_mock()
        response2 = api_client.post(
            "/api/query",
            json={
                "query": "Tell me more about neural networks",
                "session_id": session_id
            }
        )
        assert response2.status_code == status.HTTP_200_OK
        assert response2.json()["session_id"] == session_id
        mock_rag_system.session_manager.create_session.assert_not_called()

        # Clear session
        response3 = api_client.post(
            "/api/session/clear",
            json={"session_id": session_id}
        )
        assert response3.status_code == status.HTTP_200_OK
        assert response3.json()["success"] is True

    def test_get_courses_then_query(self, api_client, mock_rag_system):
        """
        Test workflow: check available courses then query.

        Validates:
        - Get course list first
        - Use course information in query
        - Both endpoints work independently
        """
        # Get available courses
        courses_response = api_client.get("/api/courses")
        assert courses_response.status_code == status.HTTP_200_OK
        courses = courses_response.json()
        assert courses["total_courses"] > 0

        # Query about a course
        query_response = api_client.post(
            "/api/query",
            json={"query": f"Tell me about {courses['course_titles'][0]}"}
        )
        assert query_response.status_code == status.HTTP_200_OK
        assert "answer" in query_response.json()
