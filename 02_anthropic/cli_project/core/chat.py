from anthropic.types import MessageParam

from core.claude import Claude
from core.tools import ToolManager
from mcp_client import MCPClient


class Chat:
    """Manages a multi-turn conversation with Claude, including tool use loops."""

    def __init__(self, claude_service: Claude, clients: dict[str, MCPClient]):
        self.claude_service: Claude = claude_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[MessageParam] = []

    async def _process_query(self, query: str):
        """Append the user query to the conversation history."""
        self.messages.append({"role": "user", "content": query})

    async def run(
        self,
        query: str,
    ) -> str:
        """Send a query and run the agentic loop until Claude returns a final answer.

        Handles tool use automatically: when Claude requests a tool, the tool is
        executed and the result is fed back until a text stop reason is received.
        """
        final_text_response = ""

        await self._process_query(query)

        while True:
            response = self.claude_service.chat(
                messages=self.messages,
                tools=await ToolManager.get_all_tools(self.clients),
            )

            self.claude_service.add_assistant_message(self.messages, response)

            if response.stop_reason == "tool_use":
                print(self.claude_service.text_from_message(response))
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )

                self.claude_service.add_user_message(self.messages, tool_result_parts)
            else:
                final_text_response = self.claude_service.text_from_message(response)
                break

        return final_text_response
