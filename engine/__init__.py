"""Ultra-Low Resource LLM Context Compression Engine.

Pipeline stages, in order:

    2. chunker      - split raw context into semantic units
    3. redundancy   - collapse near-duplicate chunks           (stage 3)
    4. density      - score each chunk's information value     (stage 4)
    5. selector     - budget-constrained greedy selection      (stage 5)
    6. abstractive  - optional local-LLM paraphrase            (stage 6)
    7. reconstruct  - stitch back into a prompt with markers   (stage 7)

Stages 3-7 are added in later commits; the chunker and the shared
config/tokenizer/type layer are complete.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, get_config, load_config
from .tokenizer import Tokenizer, get_tokenizer
from .types import Chunk, ChunkKind, StageMetrics, StageStatus

__all__ = [
    "Chunk",
    "ChunkKind",
    "Config",
    "StageMetrics",
    "StageStatus",
    "Tokenizer",
    "get_config",
    "get_tokenizer",
    "load_config",
    "__version__",
]
