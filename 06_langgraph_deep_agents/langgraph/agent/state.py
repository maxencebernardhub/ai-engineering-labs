from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # Non-empty when the graph is interrupted waiting for human input.
    # Holds a description of what the human must confirm or adjust.
    hitl_pending: str | None
