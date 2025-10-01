"""
Tests for AIGenerator tool calling functionality
"""

import os
import sys
from unittest.mock import Mock, call, patch

import pytest

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

    def test_generate_response_with_tools_no_tool_use(
        self, ai_generator_with_mock, tool_manager
    ):
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
            tool_manager=tool_manager,
        )

        assert result == "Response using general knowledge"
        # Verify tools were provided in API call
        call_args = ai_generator_with_mock.client.messages.create.call_args
        assert "tools" in call_args[1]
        assert "tool_choice" in call_args[1]

    def test_generate_response_with_tool_execution(
        self, ai_generator_with_mock, tool_manager
    ):
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
        final_response.content = [
            Mock(text="Based on search results: Machine learning is...")
        ]
        final_response.stop_reason = "end_turn"

        # Set up the client to return different responses for different calls
        ai_generator_with_mock.client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Mock tool execution
        tool_manager.execute_tool = Mock(
            return_value="Search results about machine learning"
        )

        result = ai_generator_with_mock.generate_response(
            query="What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert result == "Based on search results: Machine learning is..."

        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="machine learning basics"
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
            query="Tell me more", conversation_history=history
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

    def test_sequential_tool_execution_message_flow(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test proper message sequence construction during sequential tool execution workflow.

        Purpose:
            Validates that the sequential tool execution correctly constructs
            the conversation flow between user, assistant tool use, and tool results.
            Proper message flow is essential for Claude to understand tool execution context.

        Expected Behavior:
            Should create proper message sequence and handle tool execution with
            correct tool_use_id and content structure in the new sequential system.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager for tool execution simulation.
        """
        # Mock initial response with tool use
        initial_response = Mock()
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.input = {"query": "machine learning"}
        tool_content.id = "tool_456"
        initial_response.content = [tool_content]
        initial_response.stop_reason = "tool_use"

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Final response with tool results")]
        final_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Mock tool execution
        tool_manager.execute_tool = Mock(return_value="ML search results")

        result = ai_generator_with_mock.generate_response(
            query="What is ML?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert result == "Final response with tool results"

        # Verify API call sequence
        call_args_list = ai_generator_with_mock.client.messages.create.call_args_list
        assert len(call_args_list) == 2

        # Verify final API call structure
        final_call = call_args_list[1]
        messages = final_call[1]["messages"]

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

    def test_sequential_multiple_tool_calls_single_round(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test execution of multiple tools requested simultaneously by Claude in single round.

        Purpose:
            Ensures sequential implementation can handle complex queries where Claude decides to use
            multiple tools in a single round (e.g., searching content AND getting course outline).
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
        initial_response.stop_reason = "tool_use"

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Combined response")]
        final_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Mock tool executions
        def mock_execute_tool(tool_name, **_kwargs):
            if tool_name == "search_course_content":
                return "Search result 1"
            elif tool_name == "get_course_outline":
                return "Outline result 1"
            return "Unknown tool"

        tool_manager.execute_tool = Mock(side_effect=mock_execute_tool)

        result = ai_generator_with_mock.generate_response(
            query="Test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert result == "Combined response"

        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="first search"
        )
        tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="Test Course"
        )

        # Verify tool results structure
        call_args_list = ai_generator_with_mock.client.messages.create.call_args_list
        final_call = call_args_list[1]
        tool_results = final_call[1]["messages"][2]["content"]
        assert len(tool_results) == 2

    def test_sequential_tool_execution_error_handling(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test graceful handling of tool execution failures or errors in sequential system.

        Purpose:
            Validates that the sequential system remains stable when tools fail to execute properly
            (e.g., database errors, network issues). The AI should still provide a
            meaningful response rather than crashing or hanging.

        Expected Behavior:
            Should continue with the workflow even when tool execution returns error
            messages, passing the error information to Claude for appropriate handling
            in the final response.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager with execute_tool method
                configured to throw exceptions.
        """
        # Mock response with tool use
        initial_response = Mock()
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.input = {"query": "test"}
        tool_content.id = "tool_error"
        initial_response.content = [tool_content]
        initial_response.stop_reason = "tool_use"

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Error handled gracefully")]
        final_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Mock tool execution to raise error
        tool_manager.execute_tool = Mock(
            side_effect=Exception("Database connection failed")
        )

        result = ai_generator_with_mock.generate_response(
            query="Test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Should still return a response even if tool execution has issues
        assert result == "Error handled gracefully"

        # Verify error was handled and included in context
        call_args_list = ai_generator_with_mock.client.messages.create.call_args_list
        final_call = call_args_list[1]
        messages = final_call[1]["messages"]
        tool_result_message = messages[2]["content"][0]
        assert "Tool execution failed" in tool_result_message["content"]

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
            query="Test", tools=tools, tool_manager=None
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
            query="Follow-up question", conversation_history=history
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
            query="Test", conversation_history=None
        )

        call_args = ai_generator_with_mock.client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" not in system_content

        # Test with empty string
        ai_generator_with_mock.generate_response(query="Test", conversation_history="")

        call_args = ai_generator_with_mock.client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" not in system_content


class TestAIGeneratorSequentialToolCalling:
    """Test cases for AIGenerator sequential tool calling functionality"""

    def test_sequential_tool_calling_two_rounds_max(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test that AI generator enforces maximum of 2 sequential tool call rounds.

        Purpose:
            Validates that the system respects the 2-round maximum limit and makes a final
            API call without tools when the limit is reached, ensuring system termination.

        Expected Behavior:
            Should execute 2 rounds of tool calling, then make a final call without tools
            regardless of whether Claude wants to continue with more tools.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager for tool execution simulation.
        """
        # Round 1: Tool use response
        round1_response = Mock()
        round1_tool = Mock()
        round1_tool.type = "tool_use"
        round1_tool.name = "search_course_content"
        round1_tool.input = {"query": "machine learning"}
        round1_tool.id = "tool_round1"
        round1_response.content = [round1_tool]
        round1_response.stop_reason = "tool_use"

        # Round 2: Another tool use response
        round2_response = Mock()
        round2_tool = Mock()
        round2_tool.type = "tool_use"
        round2_tool.name = "get_course_outline"
        round2_tool.input = {"course_title": "ML Course"}
        round2_tool.id = "tool_round2"
        round2_response.content = [round2_tool]
        round2_response.stop_reason = "tool_use"

        # Final response (forced by 2-round limit)
        final_response = Mock()
        final_response.content = [Mock(text="Final answer after 2 rounds")]
        final_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        tool_manager.execute_tool = Mock(
            side_effect=["Round 1 search results", "Round 2 outline results"]
        )

        result = ai_generator_with_mock.generate_response(
            query="What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
            max_rounds=2,
        )

        assert result == "Final answer after 2 rounds"
        assert ai_generator_with_mock.client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2

    def test_sequential_tool_calling_early_termination_no_tool_use(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test termination when Claude responds without tool use blocks.

        Purpose:
            Ensures the system correctly terminates when Claude decides not to use
            additional tools, even when more rounds are available.

        Expected Behavior:
            Should stop after first round when Claude provides direct response
            without tool use, not continuing to max rounds.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager for tool execution simulation.
        """
        # Round 1: Tool use response
        round1_response = Mock()
        round1_tool = Mock()
        round1_tool.type = "tool_use"
        round1_tool.name = "search_course_content"
        round1_tool.input = {"query": "AI basics"}
        round1_tool.id = "tool_round1"
        round1_response.content = [round1_tool]
        round1_response.stop_reason = "tool_use"

        # Round 2: Direct response without tools
        round2_response = Mock()
        round2_response.content = [Mock(text="Direct answer without more tools")]
        round2_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.side_effect = [
            round1_response,
            round2_response,
        ]

        tool_manager.execute_tool = Mock(return_value="Search results about AI")

        result = ai_generator_with_mock.generate_response(
            query="What is AI?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
            max_rounds=2,
        )

        assert result == "Direct answer without more tools"
        assert ai_generator_with_mock.client.messages.create.call_count == 2
        assert tool_manager.execute_tool.call_count == 1

    def test_sequential_tool_calling_single_round_completion(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test completion when only one round of tools is sufficient.

        Purpose:
            Validates normal single-round behavior still works with the new sequential
            implementation, ensuring backward compatibility.

        Expected Behavior:
            Should complete successfully with single tool use and not attempt
            additional rounds when Claude provides complete answer.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager for tool execution simulation.
        """
        # Round 1: Tool use response
        initial_response = Mock()
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.input = {"query": "neural networks"}
        tool_content.id = "tool_single"
        initial_response.content = [tool_content]
        initial_response.stop_reason = "tool_use"

        # Follow-up: Complete answer
        final_response = Mock()
        final_response.content = [
            Mock(text="Neural networks are computational models...")
        ]
        final_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        tool_manager.execute_tool = Mock(return_value="Neural network content")

        result = ai_generator_with_mock.generate_response(
            query="What are neural networks?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert result == "Neural networks are computational models..."
        assert ai_generator_with_mock.client.messages.create.call_count == 2
        assert tool_manager.execute_tool.call_count == 1

    def test_sequential_conversation_context_preservation(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test that conversation context is maintained across sequential rounds.

        Purpose:
            Validates that message history grows correctly through rounds and that
            tool results from previous rounds remain accessible to subsequent rounds.

        Expected Behavior:
            Message chain should grow incrementally: [user] → [user, assistant, user]
            → [user, assistant, user, assistant, user] across multiple rounds.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager for tool execution simulation.
        """
        # Round 1: Tool use
        round1_response = Mock()
        round1_tool = Mock()
        round1_tool.type = "tool_use"
        round1_tool.name = "get_course_outline"
        round1_tool.input = {"course_title": "Python Course"}
        round1_tool.id = "tool_context1"
        round1_response.content = [round1_tool]
        round1_response.stop_reason = "tool_use"

        # Round 2: Another tool use
        round2_response = Mock()
        round2_tool = Mock()
        round2_tool.type = "tool_use"
        round2_tool.name = "search_course_content"
        round2_tool.input = {"query": "Python lesson 3"}
        round2_tool.id = "tool_context2"
        round2_response.content = [round2_tool]
        round2_response.stop_reason = "tool_use"

        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="Context-aware response")]
        final_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        tool_manager.execute_tool = Mock(
            side_effect=["Python course outline", "Lesson 3 content"]
        )

        result = ai_generator_with_mock.generate_response(
            query="Tell me about lesson 3 of Python course",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert result == "Context-aware response"

        # Verify message chain construction
        call_args_list = ai_generator_with_mock.client.messages.create.call_args_list

        # Initial call: 1 message (user query)
        initial_messages = call_args_list[0][1]["messages"]
        assert len(initial_messages) == 1

        # Round 2 call: All messages from sequential rounds (user + assistant + user + assistant + user)
        round2_messages = call_args_list[1][1]["messages"]
        assert len(round2_messages) == 5  # All accumulated messages

        # Final call: All messages plus final interaction
        final_messages = call_args_list[2][1]["messages"]
        assert (
            len(final_messages) == 5
        )  # Same as round 2 since it was the final call without tools

    def test_sequential_tool_execution_failure_handling(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test graceful handling of tool execution failures during sequential calls.

        Purpose:
            Validates that the system remains stable when tools fail to execute properly
            and continues the workflow with error context instead of crashing.

        Expected Behavior:
            Should include tool execution error in the context and allow Claude to
            provide a meaningful response acknowledging the failure.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager configured to fail tool execution.
        """
        # Round 1: Tool use response
        round1_response = Mock()
        round1_tool = Mock()
        round1_tool.type = "tool_use"
        round1_tool.name = "search_course_content"
        round1_tool.input = {"query": "broken search"}
        round1_tool.id = "tool_fail"
        round1_response.content = [round1_tool]
        round1_response.stop_reason = "tool_use"

        # Round 2: Response acknowledging error
        round2_response = Mock()
        round2_response.content = [
            Mock(text="I encountered an error but can help with general information")
        ]
        round2_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.side_effect = [
            round1_response,
            round2_response,
        ]

        # Mock tool execution to raise error
        tool_manager.execute_tool = Mock(
            side_effect=Exception("Database connection failed")
        )

        result = ai_generator_with_mock.generate_response(
            query="Search for something",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert result == "I encountered an error but can help with general information"
        assert ai_generator_with_mock.client.messages.create.call_count == 2

        # Verify error was included in context
        round2_call = ai_generator_with_mock.client.messages.create.call_args_list[1]
        messages = round2_call[1]["messages"]
        tool_result_message = messages[2]["content"][0]
        assert "Tool execution failed" in tool_result_message["content"]

    def test_sequential_multiple_tools_per_round(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test handling of multiple tool calls within each sequential round.

        Purpose:
            Validates that the system can handle complex scenarios where Claude
            requests multiple tools in a single round, then continues with additional
            rounds based on those results.

        Expected Behavior:
            Should execute all tools within each round and preserve all results
            in the conversation context for subsequent rounds.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager for multiple tool execution.
        """
        # Round 1: Multiple tool use
        round1_response = Mock()
        tool1 = Mock()
        tool1.type = "tool_use"
        tool1.name = "search_course_content"
        tool1.input = {"query": "machine learning"}
        tool1.id = "tool_multi1"

        tool2 = Mock()
        tool2.type = "tool_use"
        tool2.name = "get_course_outline"
        tool2.input = {"course_title": "ML Course"}
        tool2.id = "tool_multi2"

        round1_response.content = [tool1, tool2]
        round1_response.stop_reason = "tool_use"

        # Round 2: Final response
        round2_response = Mock()
        round2_response.content = [Mock(text="Combined response from multiple tools")]
        round2_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.side_effect = [
            round1_response,
            round2_response,
        ]

        def mock_execute_tool(tool_name, **_kwargs):
            if tool_name == "search_course_content":
                return "ML search results"
            elif tool_name == "get_course_outline":
                return "ML course outline"
            return "Unknown tool result"

        tool_manager.execute_tool = Mock(side_effect=mock_execute_tool)

        result = ai_generator_with_mock.generate_response(
            query="Compare ML content with course structure",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert result == "Combined response from multiple tools"
        assert tool_manager.execute_tool.call_count == 2

        # Verify both tools were executed
        tool_calls = tool_manager.execute_tool.call_args_list
        assert tool_calls[0] == call("search_course_content", query="machine learning")
        assert tool_calls[1] == call("get_course_outline", course_title="ML Course")

    def test_sequential_tool_calling_no_tools_needed(self, ai_generator_with_mock):
        """Test behavior when Claude decides no tools are needed initially.

        Purpose:
            Ensures the sequential implementation doesn't interfere with direct
            responses when no tools are required, maintaining system efficiency.

        Expected Behavior:
            Should return direct response with only 1 API call when Claude
            determines tools are not needed for the query.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
        """
        # Direct response without tools
        direct_response = Mock()
        direct_response.content = [Mock(text="Direct answer without tools")]
        direct_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.return_value = direct_response

        result = ai_generator_with_mock.generate_response(
            query="What is 2+2?",
            tools=[{"name": "test_tool", "description": "Test"}],
            tool_manager=Mock(),
        )

        assert result == "Direct answer without tools"
        assert ai_generator_with_mock.client.messages.create.call_count == 1

    def test_sequential_max_rounds_parameter_customization(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test customization of max_rounds parameter for different workflows.

        Purpose:
            Validates that the max_rounds parameter can be customized and that
            the system respects different round limits appropriately.

        Expected Behavior:
            Should enforce the specified max_rounds limit and terminate exactly
            when that limit is reached, regardless of Claude's tool requests.

        Args:
            ai_generator_with_mock: Fixture providing AIGenerator with mocked client.
            tool_manager: Fixture providing ToolManager for tool execution simulation.
        """
        # Single round limit test
        round1_response = Mock()
        round1_tool = Mock()
        round1_tool.type = "tool_use"
        round1_tool.name = "search_course_content"
        round1_tool.input = {"query": "single round"}
        round1_tool.id = "single_round"
        round1_response.content = [round1_tool]
        round1_response.stop_reason = "tool_use"

        final_response = Mock()
        final_response.content = [Mock(text="Single round complete")]
        final_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.side_effect = [
            round1_response,
            final_response,
        ]

        tool_manager.execute_tool = Mock(return_value="Single round result")

        result = ai_generator_with_mock.generate_response(
            query="Test single round",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
            max_rounds=1,
        )

        assert result == "Single round complete"
        assert ai_generator_with_mock.client.messages.create.call_count == 2
        assert tool_manager.execute_tool.call_count == 1
