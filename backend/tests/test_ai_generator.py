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
        """Test basic response generation without tool calling functionality.

        Purpose:
            Validates that AIGenerator can generate responses using only Claude's built-in
            knowledge without requiring any external tools. This is essential for handling
            general queries that don't need course-specific search functionality.

        Expected Behavior:
            Should generate a direct response from Claude without invoking any tools,
            and the API call should not include tools or tool_choice parameters.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator instance with mocked
                Anthropic client for isolated testing without real API calls.
        """
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
        """Test response generation with tools available but Claude chooses not to use them.

        Purpose:
            Ensures AIGenerator correctly handles scenarios where tools are available
            but Claude determines the query can be answered without tool execution.
            This validates the autonomous decision-making capability of the AI.

        Expected Behavior:
            Should provide tools to Claude in the API call but receive a direct response
            without tool use. The response should contain general knowledge answers.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager with registered search tools
                for course content and outline retrieval.
        """
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
        """Test complete end-to-end tool calling workflow from request to final response.

        Purpose:
            Validates the core RAG functionality where Claude autonomously decides to
            use tools, executes them, and incorporates results into the final response.
            This is critical for course-specific queries requiring knowledge retrieval.

        Expected Behavior:
            Should make initial API call, receive tool use response, execute the tool,
            make second API call with tool results, and return final synthesized response.
            Total of 2 API calls should be made in sequence.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client
                configured to simulate tool use workflow.
            tool_manager: Fixture providing ToolManager that can execute the
                search_course_content tool and return mock results.
        """
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
        """Test response generation incorporating previous conversation context.

        Purpose:
            Ensures AIGenerator properly includes conversation history in system prompts
            to maintain context across multi-turn conversations. This is essential for
            coherent dialogue and follow-up questions in the chatbot interface.

        Expected Behavior:
            Should embed the provided conversation history into the system prompt
            and generate contextually appropriate responses that reference prior exchanges.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            The method uses a mock conversation history string containing
            "User: Hello\nAssistant: Hi there!\nUser: What is AI?" as input.
        """
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
        """Test that system prompt contains all required components for tool-enabled AI.

        Purpose:
            Validates that the system prompt includes all necessary instructions for
            tool usage, ensuring Claude understands available tools and usage guidelines.
            This is crucial for reliable tool calling behavior.

        Expected Behavior:
            System prompt should contain references to available tools (search_course_content,
            get_course_outline), tool usage guidelines, and response protocols.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            The method tests with a simple "Test query" input to focus on prompt structure.
        """
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
        """Test that Anthropic API parameters are correctly configured for consistent behavior.

        Purpose:
            Ensures AIGenerator uses the correct model, temperature, token limits, and
            message structure for optimal performance. Incorrect parameters could lead
            to unreliable responses or API errors.

        Expected Behavior:
            Should use claude-sonnet-4-20250514 model, temperature=0 for deterministic
            responses, max_tokens=800 for concise answers, and proper message formatting.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            The method uses "Test query" as input to verify parameter configuration.
        """
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
        """Test proper message sequence construction during tool execution workflow.

        Purpose:
            Validates that the internal _handle_tool_execution method correctly constructs
            the conversation flow between user, assistant tool use, and tool results.
            Proper message flow is essential for Claude to understand tool execution context.

        Expected Behavior:
            Should create 3-message sequence: original user message, assistant tool use
            message, and user tool result message. Tool results must include proper
            tool_use_id and content structure.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager for tool execution simulation.
            The method uses base_params dict with predefined API parameters and
            initial_response mock containing tool use content.
        """
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
        """Test execution of multiple tools requested simultaneously by Claude.

        Purpose:
            Ensures AIGenerator can handle complex queries where Claude decides to use
            multiple tools in parallel (e.g., searching content AND getting course outline).
            This validates the system's ability to handle sophisticated query patterns.

        Expected Behavior:
            Should execute all requested tools in sequence, collect all results,
            and include them in the tool results message. Each tool should be called
            with its specific parameters.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager with mock execute_tool method
                that returns different results based on tool name (search_course_content
                vs get_course_outline).
        """
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
        """Test graceful handling of tool execution failures or errors.

        Purpose:
            Validates that the system remains stable when tools fail to execute properly
            (e.g., database errors, network issues). The AI should still provide a
            meaningful response rather than crashing or hanging.

        Expected Behavior:
            Should continue with the workflow even when tool execution returns error
            messages, passing the error information to Claude for appropriate handling
            in the final response.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager with execute_tool method
                configured to return "Tool execution failed" message instead of
                throwing exceptions.
        """
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
        """Test fallback behavior when tools are defined but no tool manager is available.

        Purpose:
            Ensures AIGenerator handles edge cases where tool definitions exist but
            the tool execution system is unavailable. This prevents crashes in
            misconfigured or partially initialized systems.

        Expected Behavior:
            Should return the response text without attempting tool execution,
            gracefully skipping the tool execution phase when tool_manager is None.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            The method uses tools list containing a test tool definition and
            tool_manager=None to simulate the edge case scenario.
        """
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
        """Test correct integration of conversation history into system prompt structure.

        Purpose:
            Validates that conversation history is properly formatted and embedded
            in the system prompt to provide Claude with conversation context.
            Proper formatting ensures Claude can understand the conversation flow.

        Expected Behavior:
            Should include "Previous conversation:" prefix in system prompt followed
            by the provided conversation history string for context preservation.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            The method uses history string "User: First question\nAssistant: First answer"
            and query "Follow-up question" to test context integration.
        """
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
        """Test proper handling of missing or empty conversation history.

        Purpose:
            Ensures AIGenerator correctly handles edge cases where no conversation
            history is provided (new sessions) or empty history strings. System should
            not include conversation context sections in these cases.

        Expected Behavior:
            Should not include "Previous conversation:" section in system prompt
            when conversation_history is None or empty string, avoiding confusion
            for Claude about non-existent context.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            The method tests both conversation_history=None and conversation_history=""
            scenarios with "Test" query to verify clean prompt generation.
        """
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