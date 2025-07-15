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
    get_agent_prompt,
    get_database_schema_context,
    validate_config
)

__all__ = [
    "TennisConfig",
    "get_agent_prompt",
    "get_database_schema_context",
    "validate_config"
] 