import sys
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Absolute path to the project root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_memory():
    """A minimal memory dict matching the pipeline's expected shape."""
    return {
        "fact": "Production must run PostgreSQL",
        "category": "technical_preference",
        "confidence": 0.8,
        "source_turn_id": 1,
        "last_access_turn": 1,
        "access_count": 1,
        "current_importance": 0.8,
    }


@pytest.fixture
def sample_ground_truth():
    """A minimal ground-truth entry from the synthetic generator."""
    return {
        "fact": "Production must run PostgreSQL",
        "category": "technical_preference",
        "source_turn": 1,
    }
