# AgentState is defined in shared/state.py (installed package) so it can be
# imported cleanly by both agents without sys.path tricks.
from shared.state import AgentState

__all__ = ["AgentState"]
