"""Tests for VerifiedMemoryLanceDB."""

import json
import tempfile
import shutil
from pathlib import Path

import pytest

from cognitive_swarm.memory.verified_memory_lancedb import (
    VerifiedMemoryLanceDB,
    DEFAULT_LANCEDB_PATH,
)


@pytest.fixture
def temp_lancedb_path():
    """Create a temporary LanceDB path for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir) / "lancedb"
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_json_path():
    """Create a temporary JSON path for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir) / "verified_memory.json"
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def lancedb_memory(temp_lancedb_path, temp_json_path):
    """Create a VerifiedMemoryLanceDB instance for testing."""
    memory = VerifiedMemoryLanceDB(
        path=temp_json_path,
        lancedb_path=temp_lancedb_path,
        similarity_threshold=0.75,
    )
    yield memory
    memory.clear()


class TestVerifiedMemoryLanceDB:
    """Test cases for VerifiedMemoryLanceDB."""

    def test_initialization(self, lancedb_memory):
        """Test that memory initializes correctly."""
        assert lancedb_memory._table is not None
        assert lancedb_memory._embedder is not None
        assert lancedb_memory.lancedb_path.exists()
        stats = lancedb_memory.get_stats()
        assert stats["total_entries"] == 0
        assert "lancedb_path" in stats

    def test_remember_basic(self, lancedb_memory):
        """Test basic remember operation."""
        lancedb_memory.remember(
            question="What is 2+2?",
            answer="4",
            tier="calculator",
            confidence=1.0,
            reliability=1.0,
        )
        
        stats = lancedb_memory.get_stats()
        assert stats["total_entries"] == 1
        assert stats["by_tier"].get("calculator", 0) == 1

    def test_remember_with_provenance(self, lancedb_memory):
        """Test remember with full provenance metadata."""
        lancedb_memory.remember(
            question="What is the capital of France?",
            answer="Paris",
            tier="retrieval",
            confidence=0.95,
            reliability=0.9,
            sources=[{"name": "wikipedia", "reliability": 0.8}],
            disagreement=[],
            source_type="encyclopedia",
            year=2024,
            journal="Wikipedia",
        )
        
        result = lancedb_memory.lookup("What is the capital of France?")
        assert result is not None
        assert result["answer"] == "Paris"
        assert result["tier"] == "retrieval"
        assert result["source_type"] == "encyclopedia"
        assert result["year"] == 2024
        assert result["journal"] == "Wikipedia"

    def test_exact_lookup(self, lancedb_memory):
        """Test exact key lookup."""
        lancedb_memory.remember("What is 2+2?", "4", tier="calculator")
        
        result = lancedb_memory.lookup("What is 2+2?")
        assert result is not None
        assert result["answer"] == "4"

    def test_code_signature_lookup(self, lancedb_memory):
        """Test code-aware lookup prevents shadowing."""
        lancedb_memory.remember(
            question="What does print(2+3) output?",
            answer="5",
            tier="executor",
        )
        lancedb_memory.remember(
            question="What does print(3+4) output?",
            answer="7",
            tier="executor",
        )
        
        result1 = lancedb_memory.lookup("What does print(2+3) output?")
        result2 = lancedb_memory.lookup("What does print(3+4) output?")
        
        assert result1["answer"] == "5"
        assert result2["answer"] == "7"
        assert result1["answer"] != result2["answer"]

    def test_string_op_signature_lookup(self, lancedb_memory):
        """Test string-op-aware lookup."""
        lancedb_memory.remember(
            question="Spell strawberry backwards",
            answer="yrrebwarts",
            tier="string-op",
        )
        lancedb_memory.remember(
            question="Spell mississippi backwards",
            answer="ippississim",
            tier="string-op",
        )
        
        result1 = lancedb_memory.lookup("Spell strawberry backwards")
        result2 = lancedb_memory.lookup("Spell mississippi backwards")
        
        assert result1["answer"] == "yrrebwarts"
        assert result2["answer"] == "ippississim"

    def test_math_op_signature_lookup(self, lancedb_memory):
        """Test math-op-aware lookup prevents operand confusion."""
        lancedb_memory.remember(
            question="What is 2 to the power of 6?",
            answer="64",
            tier="math",
        )
        lancedb_memory.remember(
            question="What is 6 squared?",
            answer="36",
            tier="math",
        )
        
        result1 = lancedb_memory.lookup("What is 2 to the power of 6?")
        result2 = lancedb_memory.lookup("What is 6 squared?")
        
        assert result1["answer"] == "64"
        assert result2["answer"] == "36"

    def test_reasoning_op_signature_lookup(self, lancedb_memory):
        """Test reasoning-op-aware lookup."""
        lancedb_memory.remember(
            question="8 people handshake how many",
            answer="28",
            tier="reasoning",
        )
        lancedb_memory.remember(
            question="100 people handshake how many",
            answer="4950",
            tier="reasoning",
        )
        
        result1 = lancedb_memory.lookup("8 people handshake how many")
        result2 = lancedb_memory.lookup("100 people handshake how many")
        
        assert result1["answer"] == "28"
        assert result2["answer"] == "4950"

    def test_vector_lookup_fallback(self, lancedb_memory):
        """Test vector search fallback for similar questions."""
        lancedb_memory.remember(
            question="What is the capital of France?",
            answer="Paris",
            tier="retrieval",
        )
        
        result = lancedb_memory._vector_lookup("Capital city of France")
        assert result is not None
        assert result["answer"] == "Paris"

    def test_vector_lookup_with_tier_filter(self, lancedb_memory):
        """Test vector lookup with tier filter."""
        lancedb_memory.remember("What is 2+2?", "4", tier="calculator")
        lancedb_memory.remember("What is 3+3?", "6", tier="math")
        
        results = lancedb_memory.lookup_with_vector(
            "What is 2+2?",
            tier_filter=["calculator"],
            k=5,
        )
        
        assert len(results) == 1
        assert results[0]["tier"] == "calculator"
        assert results[0]["answer"] == "4"

    def test_vector_lookup_with_metadata_filter(self, lancedb_memory):
        """Test vector lookup with metadata filter."""
        lancedb_memory.remember(
            question="EGFR inhibitor study",
            answer="Osimertinib shows efficacy",
            tier="retrieval",
            source_type="paper",
            year=2023,
        )
        lancedb_memory.remember(
            question="EGFR mutation frequency",
            answer="15% in lung adenocarcinoma",
            tier="retrieval",
            source_type="database",
            year=2022,
        )
        
        results = lancedb_memory.lookup_with_vector(
            "EGFR",
            metadata_filter={"source_type": "paper"},
            k=5,
        )
        
        assert len(results) == 1
        assert results[0]["source_type"] == "paper"
        assert results[0]["year"] == 2023

    def test_search_by_metadata(self, lancedb_memory):
        """Test pure metadata filter search."""
        lancedb_memory.remember(
            question="Paper 1",
            answer="Answer 1",
            tier="retrieval",
            source_type="paper",
            year=2023,
            journal="Nature",
        )
        lancedb_memory.remember(
            question="Paper 2",
            answer="Answer 2",
            tier="retrieval",
            source_type="paper",
            year=2022,
            journal="Science",
        )
        lancedb_memory.remember(
            question="DB entry",
            answer="Answer 3",
            tier="retrieval",
            source_type="database",
            year=2023,
        )
        
        results = lancedb_memory.search_by_metadata(
            {"source_type": "paper", "year": 2023},
            limit=10,
        )
        
        assert len(results) == 1
        assert results[0]["journal"] == "Nature"

    def test_hits_increment(self, lancedb_memory):
        """Test that hits increment on lookup."""
        lancedb_memory.remember("What is 2+2?", "4", tier="calculator")
        
        # First lookup (exact match) - hits becomes 1
        result = lancedb_memory.lookup("What is 2+2?")
        assert result["hits"] == 1
        
        # Second lookup - hits becomes 2
        result = lancedb_memory.lookup("What is 2+2?")
        assert result["hits"] == 2
        
        # Third lookup - hits becomes 3
        result = lancedb_memory.lookup("What is 2+2?")
        assert result["hits"] == 3

    def test_persistence_across_instances(self, temp_lancedb_path, temp_json_path):
        """Test that data persists across instances."""
        memory1 = VerifiedMemoryLanceDB(
            path=temp_json_path,
            lancedb_path=temp_lancedb_path,
        )
        memory1.remember("What is 2+2?", "4", tier="calculator")
        
        memory2 = VerifiedMemoryLanceDB(
            path=temp_json_path,
            lancedb_path=temp_lancedb_path,
        )
        
        result = memory2.lookup("What is 2+2?")
        assert result is not None
        assert result["answer"] == "4"
        
        memory1.clear()
        memory2.clear()

    def test_disagreement_handling(self, lancedb_memory):
        """Test that disagreement is stored and retrieved."""
        lancedb_memory.remember(
            question="When did Rome fall?",
            answer="476 CE",
            tier="retrieval",
            confidence=0.95,
            reliability=0.9,
            sources=[{"name": "Gibbon", "reliability": 0.9}],
            disagreement=[{"source": "Modern historians", "answer": "Gradual process", "reason": "No single date"}],
        )
        
        result = lancedb_memory.lookup("When did Rome fall?")
        assert result is not None
        assert len(result["disagreement"]) == 1
        assert result["disagreement"][0]["source"] == "Modern historians"

    def test_clear(self, lancedb_memory):
        """Test clearing all entries."""
        lancedb_memory.remember("Q1", "A1", tier="calculator")
        lancedb_memory.remember("Q2", "A2", tier="math")
        
        assert lancedb_memory.get_stats()["total_entries"] == 2
        
        lancedb_memory.clear()
        
        assert lancedb_memory.get_stats()["total_entries"] == 0
        assert lancedb_memory.lookup("Q1") is None

    def test_get_answer_shortcut(self, lancedb_memory):
        """Test get_answer convenience method."""
        lancedb_memory.remember("What is 2+2?", "4", tier="calculator")
        
        answer = lancedb_memory.get_answer("What is 2+2?")
        assert answer == "4"
        
        # Use a completely different question to avoid vector similarity match
        answer = lancedb_memory.get_answer("What is the capital of Japan?")
        assert answer is None

    def test_multiple_sources(self, lancedb_memory):
        """Test remember with multiple sources."""
        lancedb_memory.remember(
            question="Speed of light",
            answer="299,792,458 m/s",
            tier="retrieval",
            sources=[
                {"name": "NIST", "reliability": 1.0},
                {"name": "Wikipedia", "reliability": 0.8},
            ],
        )
        
        result = lancedb_memory.lookup("Speed of light")
        assert len(result["sources"]) == 2
        assert result["sources"][0]["name"] == "NIST"
        assert result["sources"][1]["name"] == "Wikipedia"


class TestVerifiedMemoryLanceDBIntegration:
    """Integration tests with realistic scenarios."""

    def test_scientific_paper_storage(self, lancedb_memory):
        """Test storing and retrieving scientific paper info."""
        lancedb_memory.remember(
            question="EGFR inhibitors for NSCLC",
            answer="Osimertinib is first-line for EGFR-mutated NSCLC",
            tier="retrieval",
            confidence=0.9,
            reliability=0.95,
            sources=[{"name": "NEJM 2020", "reliability": 0.95}],
            source_type="paper",
            year=2020,
            journal="NEJM",
            doi="10.1056/NEJMoa1913662",
            pmid="31961751",
        )
        
        results = lancedb_memory.search_by_metadata(
            {"source_type": "paper", "journal": "NEJM"},
            limit=10,
        )
        
        assert len(results) == 1
        assert results[0]["pmid"] == "31961751"
        assert "Osimertinib" in results[0]["answer"]

    def test_protocol_storage(self, lancedb_memory):
        """Test storing and retrieving lab protocols."""
        lancedb_memory.remember(
            question="RNA extraction protocol",
            answer="Trizol method: homogenize, phase separation, precipitate, wash, dissolve",
            tier="retrieval",
            confidence=0.95,
            reliability=0.9,
            source_type="protocol",
        )
        
        results = lancedb_memory.lookup_with_vector(
            "How to extract RNA",
            metadata_filter={"source_type": "protocol"},
            k=3,
        )
        
        assert len(results) >= 1
        assert "Trizol" in results[0]["answer"]

    def test_hybrid_exact_then_vector(self, lancedb_memory):
        """Test that exact lookup takes priority over vector."""
        lancedb_memory.remember(
            question="What is 2+2?",
            answer="4 (exact)",
            tier="calculator",
        )
        
        # Similar question that would match via vector
        lancedb_memory.remember(
            question="What is two plus two?",
            answer="4 (vector)",
            tier="calculator",
        )
        
        # Exact match should win
        result = lancedb_memory.lookup("What is 2+2?")
        assert result["answer"] == "4 (exact)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])