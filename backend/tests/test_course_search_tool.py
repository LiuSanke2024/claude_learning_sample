"""
Tests for CourseSearchTool.execute() method
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test cases for CourseSearchTool.execute() method"""

    def test_execute_basic_query_success(self, search_tool):
        """Test basic query execution with successful results"""
        result = search_tool.execute("introduction")

        assert isinstance(result, str)
        assert "introduction to machine learning" in result.lower()
        assert "[Test Course: Machine Learning Basics - Lesson 0]" in result

    def test_execute_with_course_name_filter(self, search_tool):
        """Test query execution with course name filtering"""
        result = search_tool.execute("basic concepts", course_name="Machine Learning")

        assert isinstance(result, str)
        assert "Test Course: Machine Learning Basics" in result
        # Verify the vector store search was called with resolved course name
        search_tool.store.search.assert_called_with(
            query="basic concepts", course_name="Machine Learning", lesson_number=None
        )

    def test_execute_with_lesson_number_filter(self, search_tool):
        """Test query execution with lesson number filtering"""
        result = search_tool.execute("advanced topics", lesson_number=2)

        assert isinstance(result, str)
        assert "Lesson 2" in result
        search_tool.store.search.assert_called_with(
            query="advanced topics", course_name=None, lesson_number=2
        )

    def test_execute_with_combined_filters(self, search_tool):
        """Test query execution with both course name and lesson number filters"""
        result = search_tool.execute(
            "neural networks", course_name="Test Course", lesson_number=2
        )

        assert isinstance(result, str)
        assert "Test Course: Machine Learning Basics" in result
        assert "Lesson 2" in result
        search_tool.store.search.assert_called_with(
            query="neural networks", course_name="Test Course", lesson_number=2
        )

    def test_execute_empty_results(self, search_tool):
        """Test handling of empty search results"""
        result = search_tool.execute("no results query")

        assert isinstance(result, str)
        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(self, search_tool):
        """Test empty results message includes filter information"""
        result = search_tool.execute(
            "no results query", course_name="Machine Learning", lesson_number=1
        )

        assert isinstance(result, str)
        assert (
            "No relevant content found in course 'Machine Learning' in lesson 1"
            in result
        )

    def test_execute_with_error_from_vector_store(self, error_vector_store):
        """Test handling of errors from vector store"""
        tool = CourseSearchTool(error_vector_store)
        result = tool.execute("any query")

        assert isinstance(result, str)
        assert "Search service unavailable" in result

    def test_execute_sources_tracking(self, search_tool):
        """Test that sources are properly tracked after search"""
        # Clear any existing sources
        search_tool.last_sources = []

        result = search_tool.execute("introduction")

        # Check that sources were populated
        assert len(search_tool.last_sources) > 0
        source = search_tool.last_sources[0]
        assert "text" in source
        assert "url" in source
        assert "Test Course: Machine Learning Basics - Lesson 0" in source["text"]

    def test_execute_sources_with_lesson_links(self, search_tool):
        """Test that lesson links are included in sources when available"""
        search_tool.last_sources = []

        result = search_tool.execute("introduction")

        source = search_tool.last_sources[0]
        assert source["url"] == "https://example.com/lesson1"  # From mock

    def test_execute_sources_without_lesson_number(self, mock_vector_store):
        """Test sources when no lesson number is in metadata (course-level result)"""

        # Mock search to return result without lesson_number
        def mock_search_no_lesson(
            query, course_name=None, lesson_number=None, limit=None
        ):
            return SearchResults(
                documents=["Course overview content"],
                metadata=[
                    {"course_title": "Test Course: Machine Learning Basics"}
                ],  # No lesson_number
                distances=[0.1],
            )

        mock_vector_store.search = Mock(side_effect=mock_search_no_lesson)
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("overview")

        assert len(tool.last_sources) > 0
        source = tool.last_sources[0]
        assert source["text"] == "Test Course: Machine Learning Basics"
        # Should try to get course link instead of lesson link
        mock_vector_store.get_course_link.assert_called_with(
            "Test Course: Machine Learning Basics"
        )

    def test_execute_graceful_link_failure(self, mock_vector_store):
        """Test graceful handling when link retrieval fails"""
        # Mock link methods to raise exceptions
        mock_vector_store.get_lesson_link = Mock(
            side_effect=Exception("Link service down")
        )
        mock_vector_store.get_course_link = Mock(
            side_effect=Exception("Link service down")
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("introduction")

        # Should still work, just without links
        assert isinstance(result, str)
        assert "introduction to machine learning" in result.lower()
        assert len(tool.last_sources) > 0
        source = tool.last_sources[0]
        assert source["url"] is None  # No URL due to exception

    def test_execute_result_formatting(self, search_tool):
        """Test that results are properly formatted with headers and content"""
        result = search_tool.execute("introduction")

        lines = result.split("\n")
        # Should have header format: [Course - Lesson X]
        assert any(
            "[Test Course: Machine Learning Basics - Lesson 0]" in line
            for line in lines
        )
        # Should have content
        assert any("introduction to machine learning" in line.lower() for line in lines)

    def test_execute_multiple_results_formatting(self, mock_vector_store):
        """Test formatting when multiple results are returned"""

        def mock_search_multiple(
            query, course_name=None, lesson_number=None, limit=None
        ):
            return SearchResults(
                documents=[
                    "First result about machine learning",
                    "Second result about algorithms",
                ],
                metadata=[
                    {
                        "course_title": "Test Course: Machine Learning Basics",
                        "lesson_number": 0,
                    },
                    {
                        "course_title": "Test Course: Machine Learning Basics",
                        "lesson_number": 1,
                    },
                ],
                distances=[0.1, 0.2],
            )

        mock_vector_store.search = Mock(side_effect=mock_search_multiple)
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("machine learning")

        # Should have multiple sections separated by double newlines
        sections = result.split("\n\n")
        assert len(sections) == 2
        assert "[Test Course: Machine Learning Basics - Lesson 0]" in sections[0]
        assert "[Test Course: Machine Learning Basics - Lesson 1]" in sections[1]

    def test_execute_tool_definition_structure(self, search_tool):
        """Test that tool definition has correct structure for Anthropic API"""
        tool_def = search_tool.get_tool_definition()

        assert tool_def["name"] == "search_course_content"
        assert "description" in tool_def
        assert "input_schema" in tool_def

        schema = tool_def["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        properties = schema["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

        # Query should be required
        assert "query" in schema["required"]
        assert "course_name" not in schema["required"]
        assert "lesson_number" not in schema["required"]

    def test_execute_parameter_validation(self, search_tool):
        """Test that execute handles different parameter types correctly"""
        # Test with string lesson number (should work if converted)
        result = search_tool.execute("test", lesson_number=1)
        assert isinstance(result, str)

        # Test with None values
        result = search_tool.execute("test", course_name=None, lesson_number=None)
        assert isinstance(result, str)

        # Test with empty strings
        result = search_tool.execute("test", course_name="", lesson_number=None)
        assert isinstance(result, str)

    def test_execute_sources_reset_between_calls(self, search_tool):
        """Test that sources are properly reset between different search calls"""
        # First search
        search_tool.execute("introduction")
        first_sources = search_tool.last_sources.copy()

        # Second search
        search_tool.execute("neural networks")
        second_sources = search_tool.last_sources

        # Sources should be different (assuming different mock responses)
        assert first_sources != second_sources
        # But both should have content
        assert len(first_sources) > 0
        assert len(second_sources) > 0

    def test_execute_edge_case_empty_query(self, search_tool):
        """Test handling of edge case with empty query"""
        result = search_tool.execute("")

        # Should still work (vector store will handle empty query)
        assert isinstance(result, str)

    def test_execute_edge_case_very_long_query(self, search_tool):
        """Test handling of very long queries"""
        long_query = "machine learning " * 100  # Very long query
        result = search_tool.execute(long_query)

        assert isinstance(result, str)
        # Should pass query to vector store without modification
        search_tool.store.search.assert_called_with(
            query=long_query, course_name=None, lesson_number=None
        )
