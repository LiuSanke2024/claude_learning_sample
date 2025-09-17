"""
Tests for AIGenerator tool calling functionality
"""
import pytest
from unittest.mock import Mock, patch, call
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import ToolManager


class TestAIGeneratorToolCalling:
    """Test cases for AIGenerator tool calling functionality"""

    def test_generate_response_without_tools(self, ai_generator_with_mock):
        """Test basic response generation without tool calling"""
        # Mock response without tools
        mock_response = Mock()
        mock_response.content = [Mock(text="Simple response without tools")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        result = ai_generator_with_mock.generate_response("What is machine learning?")

        assert result == "Simple response without tools"
        # Verify API call was made without tools
        ai_generator_with_mock.client.messages.create.assert_called_once()
        call_args = ai_generator_with_mock.client.messages.create.call_args
        assert "tools" not in call_args[1]

    def test_generate_response_with_tools_no_tool_use(self, ai_generator_with_mock, tool_manager):
        """Test response generation with tools available but not used"""
        # Mock response that doesn't use tools
        mock_response = Mock()
        mock_response.content = [Mock(text="Response using general knowledge")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        result = ai_generator_with_mock.generate_response(
            query="What is AI?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        assert result == "Response using general knowledge"
        # Verify tools were provided in API call
        call_args = ai_generator_with_mock.client.messages.create.call_args
        assert "tools" in call_args[1]
        assert "tool_choice" in call_args[1]

    def test_generate_response_with_tool_execution(self, ai_generator_with_mock, tool_manager):
        """Test complete tool calling workflow"""
        # Mock initial response with tool use
        initial_response = Mock()
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.input = {"query": "machine learning basics"}
        tool_content.id = "tool_123"
        initial_response.content = [tool_content]
        initial_response.stop_reason = "tool_use"

        # Mock final response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="Based on search results: Machine learning is...")]
        final_response.stop_reason = "end_turn"

        # Set up the client to return different responses for different calls
        ai_generator_with_mock.client.messages.create.side_effect = [initial_response, final_response]

        # Mock tool execution
        tool_manager.execute_tool = Mock(return_value="Search results about machine learning")

        result = ai_generator_with_mock.generate_response(
            query="What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        assert result == "Based on search results: Machine learning is..."

        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning basics"
        )

        # Verify two API calls were made
        assert ai_generator_with_mock.client.messages.create.call_count == 2

    def test_generate_response_with_conversation_history(self, ai_generator_with_mock):
        """Test response generation with conversation history"""
        history = "User: Hello\nAssistant: Hi there!\nUser: What is AI?"

        mock_response = Mock()
        mock_response.content = [Mock(text="AI response with context")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        result = ai_generator_with_mock.generate_response(
            query="Tell me more",
            conversation_history=history
        )

        assert result == "AI response with context"

        # Verify system prompt includes conversation history
        call_args = ai_generator_with_mock.client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert history in system_content

    def test_generate_response_system_prompt_structure(self, ai_generator_with_mock):
        """Test that system prompt is properly structured"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        ai_generator_with_mock.generate_response("Test query")

        call_args = ai_generator_with_mock.client.messages.create.call_args
        system_content = call_args[1]["system"]

        # Check system prompt contains key elements
        assert "search_course_content" in system_content
        assert "get_course_outline" in system_content
        assert "Tool Usage Guidelines" in system_content
        assert "Response Protocol" in system_content

    def test_generate_response_api_parameters(self, ai_generator_with_mock):
        """Test that API parameters are correctly set"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        ai_generator_with_mock.generate_response("Test query")

        call_args = ai_generator_with_mock.client.messages.create.call_args
        params = call_args[1]

        assert params["model"] == "claude-sonnet-4-20250514"
        assert params["temperature"] == 0
        assert params["max_tokens"] == 800
        assert len(params["messages"]) == 1
        assert params["messages"][0]["role"] == "user"
        assert params["messages"][0]["content"] == "Test query"

    def test_handle_tool_execution_message_flow(self, ai_generator_with_mock, tool_manager):
        """Test the message flow during tool execution"""
        # Create initial API params
        base_params = {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": "What is ML?"}],
            "system": "System prompt here"
        }

        # Mock initial response with tool use
        initial_response = Mock()
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.input = {"query": "machine learning"}
        tool_content.id = "tool_456"
        initial_response.content = [tool_content]

        # Mock tool execution
        tool_manager.execute_tool = Mock(return_value="ML search results")

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Final response with tool results")]

        ai_generator_with_mock.client.messages.create.return_value = final_response

        result = ai_generator_with_mock._handle_tool_execution(
            initial_response, base_params, tool_manager
        )

        assert result == "Final response with tool results"

        # Verify final API call structure
        call_args = ai_generator_with_mock.client.messages.create.call_args
        messages = call_args[1]["messages"]

        # Should have: original user message, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Check tool result structure
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_456"
        assert tool_results[0]["content"] == "ML search results"

    def test_handle_multiple_tool_calls(self, ai_generator_with_mock, tool_manager):
        """Test handling of multiple tool calls in one response"""
        base_params = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Test"}],
            "system": "System prompt"
        }

        # Mock response with multiple tool uses
        initial_response = Mock()
        tool1 = Mock()
        tool1.type = "tool_use"
        tool1.name = "search_course_content"
        tool1.input = {"query": "first search"}
        tool1.id = "tool_1"

        tool2 = Mock()
        tool2.type = "tool_use"
        tool2.name = "get_course_outline"
        tool2.input = {"course_title": "Test Course"}
        tool2.id = "tool_2"

        initial_response.content = [tool1, tool2]

        # Mock tool executions
        def mock_execute_tool(tool_name, **kwargs):
            if tool_name == "search_course_content":
                return "Search result 1"
            elif tool_name == "get_course_outline":
                return "Outline result 1"
            return "Unknown tool"

        tool_manager.execute_tool = Mock(side_effect=mock_execute_tool)

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Combined response")]
        ai_generator_with_mock.client.messages.create.return_value = final_response

        result = ai_generator_with_mock._handle_tool_execution(
            initial_response, base_params, tool_manager
        )

        assert result == "Combined response"

        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("search_course_content", query="first search")
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="Test Course")

        # Verify tool results structure
        call_args = ai_generator_with_mock.client.messages.create.call_args
        tool_results = call_args[1]["messages"][2]["content"]
        assert len(tool_results) == 2

    def test_tool_execution_error_handling(self, ai_generator_with_mock, tool_manager):
        """Test handling of errors during tool execution"""
        base_params = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Test"}],
            "system": "System prompt"
        }

        # Mock response with tool use
        initial_response = Mock()
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.input = {"query": "test"}
        tool_content.id = "tool_error"
        initial_response.content = [tool_content]

        # Mock tool execution to raise error
        tool_manager.execute_tool = Mock(return_value="Tool execution failed")

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Error handled gracefully")]
        ai_generator_with_mock.client.messages.create.return_value = final_response

        result = ai_generator_with_mock._handle_tool_execution(
            initial_response, base_params, tool_manager
        )

        # Should still return a response even if tool execution has issues
        assert result == "Error handled gracefully"

    def test_no_tool_manager_provided(self, ai_generator_with_mock):
        """Test behavior when tool_manager is None but tools are provided"""
        tools = [{"name": "test_tool", "description": "Test"}]

        # Mock response with tool use but proper content structure
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [Mock(text="Tool use response")]
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        # Should not crash when tool_manager is None, but should return the response text
        result = ai_generator_with_mock.generate_response(
            query="Test",
            tools=tools,
            tool_manager=None
        )

        # Should return the response text since tool execution is skipped
        assert result == "Tool use response"

    def test_conversation_history_formatting(self, ai_generator_with_mock):
        """Test proper formatting of conversation history in system prompt"""
        history = "User: First question\nAssistant: First answer"

        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        ai_generator_with_mock.generate_response(
            query="Follow-up question",
            conversation_history=history
        )

        call_args = ai_generator_with_mock.client.messages.create.call_args
        system_content = call_args[1]["system"]

        assert "Previous conversation:" in system_content
        assert history in system_content

    def test_empty_conversation_history(self, ai_generator_with_mock):
        """Test handling of empty or None conversation history"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        # Test with None
        ai_generator_with_mock.generate_response(
            query="Test",
            conversation_history=None
        )

        call_args = ai_generator_with_mock.client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" not in system_content

        # Test with empty string
        ai_generator_with_mock.generate_response(
            query="Test",
            conversation_history=""
        )

        call_args = ai_generator_with_mock.client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" not in system_content