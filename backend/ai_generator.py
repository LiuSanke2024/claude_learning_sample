from typing import Dict, List, Optional

import anthropic


class AIGenerator:
    """
    Handles interactions with Anthropic's Claude API for generating responses.

    Features:
    - Sequential tool calling: Claude can make up to 2 rounds of tool calls per query
    - Conversation context preservation across tool execution rounds
    - Graceful error handling for tool failures
    - Backward compatible with single-round tool calling
    """

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **search_course_content** - For searching specific course content and detailed educational materials
2. **get_course_outline** - For getting complete course outlines including title, course link, and all lessons with numbers and titles

Tool Usage Guidelines:
- **Sequential tool calling**: You can use up to 2 sequential tool calls to gather comprehensive information
- **Tool reasoning**: First tool call can inform what additional information you need for a complete answer
- **Example workflow**: Get course outline first, then search for specific content within identified lessons
- **Course outline queries** (e.g., "outline of X course", "lessons in Y course"): Use get_course_outline tool
- **Content-specific questions**: Use search_course_content tool
- **General knowledge questions**: Answer using existing knowledge without tools
- **Tool combination**: You can search content first, then get outlines, or vice versa based on the query
- **Context awareness**: Consider results from previous tool calls when deciding on additional tools
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **Course outline requests**: Use get_course_outline to return course title, course link, and complete lesson list
- **Course-specific content questions**: Use search_course_content first, then follow up with additional searches if needed
- **Complex queries**: Break down into multiple tool calls if needed to gather comprehensive information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def _build_system_content(self, conversation_history: Optional[str]) -> str:
        """
        Build system content with optional conversation history.

        Args:
            conversation_history: Previous conversation messages to include in system prompt

        Returns:
            Complete system prompt content with or without conversation history
        """
        return (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

    def _make_api_call(
        self, messages: List[Dict], system_content: str, tools: Optional[List] = None
    ):
        """
        Make API call with proper parameter setup.

        Args:
            messages: List of conversation messages
            system_content: System prompt content
            tools: Optional list of tools to make available to Claude

        Returns:
            Anthropic API response object
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        return self.client.messages.create(**api_params)

    def _execute_tools(self, response, tool_manager) -> Optional[List[Dict]]:
        """
        Execute tools and return formatted results.

        Args:
            response: Anthropic API response containing tool use requests
            tool_manager: Tool manager instance to execute tools

        Returns:
            List of formatted tool results, or None if no tools were executed
        """
        tool_results = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Handle tool execution errors gracefully
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution failed: {str(e)}",
                        }
                    )

        return tool_results if tool_results else None

    def _execute_sequential_rounds(
        self,
        initial_response,
        system_content: str,
        initial_messages: List[Dict],
        tools: Optional[List],
        tool_manager,
        max_rounds: int,
    ) -> str:
        """
        Execute up to max_rounds of sequential tool calling with conversation context preservation.

        Args:
            initial_response: The initial response containing tool use requests
            system_content: System prompt content for API calls
            initial_messages: Initial message list to build upon
            tools: Available tools for subsequent rounds
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of rounds to execute

        Returns:
            Final response text after sequential tool execution
        """
        messages = initial_messages.copy()
        current_response = initial_response
        round_count = 0

        while round_count < max_rounds:
            # Check termination condition: No tool use in response
            if current_response.stop_reason != "tool_use":
                return current_response.content[0].text

            # Add AI's response to conversation
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute tools and collect results
            tool_results = self._execute_tools(current_response, tool_manager)

            if not tool_results:
                # Tool execution failed - return what we have
                return (
                    current_response.content[0].text
                    if current_response.content
                    else "Tool execution failed"
                )

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})
            round_count += 1

            # Check termination condition: Max rounds reached
            if round_count >= max_rounds:
                # Final call without tools for definitive answer
                final_response = self._make_api_call(
                    messages, system_content, tools=None
                )
                return final_response.content[0].text

            # Continue with next round - keep tools available
            current_response = self._make_api_call(messages, system_content, tools)

        # Fallback - should not reach here
        return (
            current_response.content[0].text
            if current_response.content
            else "No response generated"
        )

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with sequential tool calling capability.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of sequential tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Build system content with conversation history
        system_content = self._build_system_content(conversation_history)

        # Prepare initial messages
        initial_messages = [{"role": "user", "content": query}]

        # Get initial response from Claude
        response = self._make_api_call(initial_messages, system_content, tools)

        # Handle sequential tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._execute_sequential_rounds(
                response,
                system_content,
                initial_messages,
                tools,
                tool_manager,
                max_rounds,
            )

        # Return direct response if no tools used
        return response.content[0].text
