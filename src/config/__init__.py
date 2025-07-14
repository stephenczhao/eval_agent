"""
Tennis Intelligence System - Configuration Package
==================================================

This package contains system configuration, prompts, and settings:
- Settings: Main configuration management
- Database schema context for agents
- System prompts and templates
"""

from .settings import (
    TennisConfig,
    DATABASE_SCHEMA_CONTEXT,
    ORCHESTRATOR_PROMPT,
    SQL_AGENT_PROMPT,
    SEARCH_AGENT_PROMPT,
    SYNTHESIZER_PROMPT
)

__all__ = [
    "TennisConfig",
    "DATABASE_SCHEMA_CONTEXT",
    "ORCHESTRATOR_PROMPT",
    "SQL_AGENT_PROMPT", 
    "SEARCH_AGENT_PROMPT",
    "SYNTHESIZER_PROMPT"
] 