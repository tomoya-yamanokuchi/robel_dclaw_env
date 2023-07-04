from typing import TypedDict, Any
from .AbstractTaskSpaceTransformer import AbstractTaskSpaceTransformer


class TaskSpaceDict(TypedDict):
    """Typed User definition."""
    transformer           : AbstractTaskSpaceTransformer
    TaskSpacePosition     : Any
    TaskSpaceDiffPosition : Any
