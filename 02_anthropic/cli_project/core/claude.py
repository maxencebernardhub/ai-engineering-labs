from anthropic import Anthropic
from anthropic.types import Message


class Claude:
    """Thin wrapper around the Anthropic client for chat-oriented interactions."""

    def __init__(self, model: str):
        self.client = Anthropic()
        self.model = model

    def add_user_message(self, messages: list, message):
        """Append a user turn to the messages list.

        Accepts either a raw Message object (extracts its content) or a plain value.
        """
        user_message = {
            "role": "user",
            "content": message.content if isinstance(message, Message) else message,
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        """Append an assistant turn to the messages list.

        Accepts either a raw Message object (extracts its content) or a plain value.
        """
        assistant_message = {
            "role": "assistant",
            "content": message.content if isinstance(message, Message) else message,
        }
        messages.append(assistant_message)

    def text_from_message(self, message: Message):
        """Extract and join all text blocks from a Message response."""
        return "\n".join(
            [block.text for block in message.content if block.type == "text"]
        )

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=None,
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> Message:
        """Send a chat request to Claude and return the raw Message response.

        Args:
            messages: Conversation history as a list of MessageParam dicts.
            system: Optional system prompt.
            temperature: Sampling temperature (default 1.0).
            stop_sequences: Optional list of stop sequences.
            tools: Optional list of tool definitions to make available.
            thinking: Enable extended thinking mode (default False).
            thinking_budget: Token budget for thinking (default 1024).
        """
        if stop_sequences is None:
            stop_sequences = []
        params = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": messages,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
        }

        if thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        if tools:
            params["tools"] = tools

        if system:
            params["system"] = system

        message = self.client.messages.create(**params)
        return message
