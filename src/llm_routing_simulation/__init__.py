"""CPU-only routing simulation from a versioned LLM cache."""

CACHE_SCHEMA_VERSION = "llm-routing-cache-v1"
CACHE_SCHEMA_VERSIONS = frozenset(
    {"llm-routing-cache-v1", "llm-routing-cache-v2"}
)
