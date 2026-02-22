"""
Tests for the Regulation Citation Tree (RCT) data structure.
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.citation_tree import (
    RegulationCitationTree,
    CitationNode,
    CitationType
)


class TestCitationNode:
    """Tests for the CitationNode class."""
    
    def test_create_node(self):
        """Test creating a citation node."""
        node = CitationNode(
            id="test_1",
            type=CitationType.ARTICLE,
            reference="Article 92",
            text="Own funds requirements",
            metadata={"source": "CRR"}
        )
        
        assert node.id == "test_1"
        assert node.type == CitationType.ARTICLE
        assert node.reference == "Article 92"
        assert node.text == "Own funds requirements"
        assert node.metadata["source"] == "CRR"
    
    def test_node_serialization(self):
        """Test node to_dict and from_dict."""
        original = CitationNode(
            id="test_1",
            type=CitationType.PARAGRAPH,
            reference="Paragraph 1",
            text="Test text",
            parent_id="parent_1"
        )
        
        # Convert to dict
        data = original.to_dict()
        assert data['id'] == "test_1"
        assert data['type'] == "paragraph"
        
        # Convert back
        restored = CitationNode.from_dict(data)
        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.reference == original.reference


class TestRegulationCitationTree:
    """Tests for the RegulationCitationTree class."""
    
    @pytest.fixture
    def tree(self):
        """Create a fresh tree for each test."""
        return RegulationCitationTree(name="test_tree")
    
    @pytest.fixture
    def populated_tree(self):
        """Create a tree with sample data."""
        tree = RegulationCitationTree(name="sample_tree")
        
        # Add CRR regulation
        crr_id = tree.add_citation(
            "CRR",
            CitationType.REGULATION,
            text="Capital Requirements Regulation (EU) No 575/2013",
            metadata={"year": 2013, "type": "EU Regulation"}
        )
        
        # Add Article 92
        art92_id = tree.add_citation(
            "Article 92",
            CitationType.ARTICLE,
            text="Own funds requirements",
            parent_reference="CRR",
            metadata={"topic": "capital requirements"}
        )
        
        # Add paragraphs under Article 92
        tree.add_citation(
            "Paragraph 1",
            CitationType.PARAGRAPH,
            text="Point (a) CET1 capital ratio of 4.5%",
            parent_reference="Article 92"
        )
        
        tree.add_citation(
            "Paragraph 2",
            CitationType.PARAGRAPH,
            text="Point (b) Tier 1 capital ratio of 6%",
            parent_reference="Article 92"
        )
        
        # Add another regulation
        crd_id = tree.add_citation(
            "CRD IV",
            CitationType.DIRECTIVE,
            text="Capital Requirements Directive 2013/36/EU",
            metadata={"year": 2013}
        )
        
        return tree
    
    def test_tree_initialization(self, tree):
        """Test tree initialization."""
        assert tree.name == "test_tree"
        assert len(tree.nodes) == 0
        assert len(tree.root_ids) == 0
    
    def test_add_root_citation(self, tree):
        """Test adding a root citation."""
        node_id = tree.add_citation(
            "CRR",
            CitationType.REGULATION,
            text="Capital Requirements Regulation"
        )
        
        assert len(tree.nodes) == 1
        assert len(tree.root_ids) == 1
        assert node_id in tree.nodes
        
        node = tree.get_citation(node_id)
        assert node.reference == "CRR"
        assert node.context == "CRR"
        assert node.parent_id is None
    
    def test_add_child_citation(self, tree):
        """Test adding a child citation."""
        # Add parent
        parent_id = tree.add_citation("CRR", CitationType.REGULATION)
        
        # Add child
        child_id = tree.add_citation(
            "Article 92",
            CitationType.ARTICLE,
            parent_reference="CRR"
        )
        
        assert len(tree.nodes) == 2
        
        child = tree.get_citation(child_id)
        assert child.parent_id == parent_id
        assert child.context == "CRR > Article 92"
        
        parent = tree.get_citation(parent_id)
        assert child_id in parent.children_ids
    
    def test_add_citation_invalid_parent(self, tree):
        """Test adding citation with non-existent parent."""
        with pytest.raises(ValueError, match="Parent reference.*not found"):
            tree.add_citation(
                "Article 92",
                CitationType.ARTICLE,
                parent_reference="NonExistent"
            )
    
    def test_get_citation_by_reference(self, populated_tree):
        """Test getting citation by reference."""
        node = populated_tree.get_citation_by_reference("Article 92")
        
        assert node is not None
        assert node.reference == "Article 92"
        assert node.type == CitationType.ARTICLE
    
    def test_find_citations_by_reference(self, populated_tree):
        """Test finding all citations with a reference."""
        nodes = populated_tree.find_citations_by_reference("CRR")
        
        assert len(nodes) == 1
        assert nodes[0].reference == "CRR"
    
    def test_find_citations_by_type(self, populated_tree):
        """Test finding citations by type."""
        # Find all paragraphs
        paragraphs = populated_tree.find_citations_by_type(CitationType.PARAGRAPH)
        
        assert len(paragraphs) == 2
        assert all(p.type == CitationType.PARAGRAPH for p in paragraphs)
    
    def test_find_citations_by_keyword(self, populated_tree):
        """Test finding citations by keyword."""
        # Search for "capital"
        results = populated_tree.find_citations_by_keyword("capital")
        
        assert len(results) >= 2
        assert any("capital" in r.text.lower() for r in results)
    
    def test_find_citations_by_context(self, populated_tree):
        """Test finding citations by context pattern."""
        results = populated_tree.find_citations_by_context("CRR > Article 92")
        
        # Should find Article 92 and its children
        assert len(results) >= 3
        assert all("CRR > Article 92" in r.context for r in results)
    
    def test_get_full_citation_path(self, populated_tree):
        """Test getting full path to a citation."""
        # Find a paragraph node
        paragraphs = populated_tree.find_citations_by_type(CitationType.PARAGRAPH)
        para_id = paragraphs[0].id
        
        path = populated_tree.get_full_citation_path(para_id)
        
        assert len(path) == 3  # CRR > Article 92 > Paragraph
        assert path[0].reference == "CRR"
        assert path[1].reference == "Article 92"
        assert path[2].type == CitationType.PARAGRAPH
    
    def test_get_citation_text_with_context(self, populated_tree):
        """Test getting formatted citation text."""
        node = populated_tree.get_citation_by_reference("Article 92")
        text = populated_tree.get_citation_text_with_context(node.id, include_children=True)
        
        assert "[CRR > Article 92]" in text
        assert "Own funds requirements" in text
        assert "Sub-citations:" in text
    
    def test_get_children(self, populated_tree):
        """Test getting children of a node."""
        art92 = populated_tree.get_citation_by_reference("Article 92")
        children = populated_tree.get_children(art92.id)
        
        assert len(children) == 2
        assert all(c.type == CitationType.PARAGRAPH for c in children)
    
    def test_get_parent(self, populated_tree):
        """Test getting parent of a node."""
        para = populated_tree.find_citations_by_type(CitationType.PARAGRAPH)[0]
        parent = populated_tree.get_parent(para.id)
        
        assert parent is not None
        assert parent.reference == "Article 92"
    
    def test_get_siblings(self, populated_tree):
        """Test getting siblings of a node."""
        paragraphs = populated_tree.find_citations_by_type(CitationType.PARAGRAPH)
        siblings = populated_tree.get_siblings(paragraphs[0].id)
        
        assert len(siblings) == 1
        assert siblings[0].type == CitationType.PARAGRAPH
    
    def test_update_citation(self, populated_tree):
        """Test updating a citation."""
        node = populated_tree.get_citation_by_reference("Article 92")
        
        success = populated_tree.update_citation(
            node.id,
            text="Updated text for Article 92"
        )
        
        assert success
        
        updated = populated_tree.get_citation(node.id)
        assert updated.text == "Updated text for Article 92"
    
    def test_delete_citation_leaf(self, populated_tree):
        """Test deleting a leaf citation."""
        para = populated_tree.find_citations_by_type(CitationType.PARAGRAPH)[0]
        para_id = para.id
        
        success = populated_tree.delete_citation(para_id)
        
        assert success
        assert para_id not in populated_tree.nodes
    
    def test_delete_citation_with_children_error(self, populated_tree):
        """Test that deleting node with children raises error."""
        art92 = populated_tree.get_citation_by_reference("Article 92")
        
        with pytest.raises(ValueError, match="Cannot delete node with children"):
            populated_tree.delete_citation(art92.id)
    
    def test_delete_citation_with_children_recursive(self, populated_tree):
        """Test deleting citation with children recursively."""
        crr = populated_tree.get_citation_by_reference("CRR")
        initial_count = len(populated_tree.nodes)
        
        success = populated_tree.delete_citation(crr.id, delete_children=True)
        
        assert success
        # Should have deleted CRR, Article 92, and 2 paragraphs = 4 nodes
        # Only CRD IV should remain
        assert len(populated_tree.nodes) == 1
    
    def test_get_statistics(self, populated_tree):
        """Test getting tree statistics."""
        stats = populated_tree.get_statistics()
        
        assert stats['total_nodes'] == 5
        assert stats['root_nodes'] == 2
        assert stats['nodes_by_type']['regulation'] == 1
        assert stats['nodes_by_type']['directive'] == 1
        assert stats['nodes_by_type']['article'] == 1
        assert stats['nodes_by_type']['paragraph'] == 2
        assert stats['max_depth'] >= 2
    
    def test_save_and_load(self, populated_tree):
        """Test saving and loading the tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_tree.json"
            
            # Save
            populated_tree.save(str(save_path))
            assert save_path.exists()
            
            # Load into new tree
            new_tree = RegulationCitationTree(name="loaded_tree")
            new_tree.load(str(save_path))
            
            # Verify
            assert new_tree.name == populated_tree.name
            assert len(new_tree.nodes) == len(populated_tree.nodes)
            assert len(new_tree.root_ids) == len(populated_tree.root_ids)
            
            # Check a specific node
            node = new_tree.get_citation_by_reference("Article 92")
            assert node is not None
            assert node.reference == "Article 92"
    
    def test_export_markdown(self, populated_tree):
        """Test exporting tree as Markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "tree.md"
            
            populated_tree.export_markdown(str(md_path))
            
            assert md_path.exists()
            
            content = md_path.read_text()
            assert "CRR" in content
            assert "Article 92" in content
            assert "Paragraph" in content
    
    def test_len(self, populated_tree):
        """Test len() operator."""
        assert len(populated_tree) == 5
    
    def test_repr(self, populated_tree):
        """Test string representation."""
        rep = repr(populated_tree)
        assert "sample_tree" in rep
        assert "nodes=5" in rep
        assert "roots=2" in rep


class TestCitationTreeHierarchy:
    """Tests for complex hierarchical structures."""
    
    def test_deep_hierarchy(self):
        """Test creating a deep hierarchical structure."""
        tree = RegulationCitationTree(name="deep_tree")
        
        # Build: CRR > Article 92 > Para 1 > Point (a) > Subpoint (i)
        tree.add_citation("CRR", CitationType.REGULATION)
        tree.add_citation("Article 92", CitationType.ARTICLE, parent_reference="CRR")
        tree.add_citation("Paragraph 1", CitationType.PARAGRAPH, parent_reference="Article 92")
        tree.add_citation("Point (a)", CitationType.POINT, parent_reference="Paragraph 1")
        tree.add_citation("Subpoint (i)", CitationType.SUBPOINT, parent_reference="Point (a)")
        
        # Verify path
        subpoint = tree.get_citation_by_reference("Subpoint (i)")
        path = tree.get_full_citation_path(subpoint.id)
        
        assert len(path) == 5
        assert path[0].reference == "CRR"
        assert path[4].reference == "Subpoint (i)"
        assert "CRR > Article 92 > Paragraph 1 > Point (a) > Subpoint (i)" in subpoint.context
    
    def test_multiple_regulations(self):
        """Test tree with multiple top-level regulations."""
        tree = RegulationCitationTree(name="multi_reg")
        
        # Add multiple regulations
        tree.add_citation("CRR", CitationType.REGULATION)
        tree.add_citation("CRD IV", CitationType.DIRECTIVE)
        tree.add_citation("Basel III", CitationType.REGULATION)
        
        assert len(tree.root_ids) == 3
        
        regs = tree.find_citations_by_type(CitationType.REGULATION)
        assert len(regs) == 2
    
    def test_cross_references(self):
        """Test storing cross-references in metadata."""
        tree = RegulationCitationTree(name="cross_ref")
        
        # Add citations with cross-references
        tree.add_citation("CRR", CitationType.REGULATION)
        tree.add_citation(
            "Article 92",
            CitationType.ARTICLE,
            parent_reference="CRR",
            metadata={"cross_references": ["CRD IV Article 73"]}
        )
        
        art92 = tree.get_citation_by_reference("Article 92")
        assert "cross_references" in art92.metadata
        assert "CRD IV Article 73" in art92.metadata["cross_references"]


class TestCitationTreeEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_tree_operations(self):
        """Test operations on empty tree."""
        tree = RegulationCitationTree(name="empty")
        
        assert tree.get_citation("nonexistent") is None
        assert tree.get_citation_by_reference("Article 1") is None
        assert len(tree.find_citations_by_type(CitationType.ARTICLE)) == 0
        assert len(tree.find_citations_by_keyword("test")) == 0
    
    def test_duplicate_references(self):
        """Test handling duplicate references at different levels."""
        tree = RegulationCitationTree(name="dup_test")
        
        # Add Article 1 under CRR
        tree.add_citation("CRR", CitationType.REGULATION)
        tree.add_citation("Article 1", CitationType.ARTICLE, parent_reference="CRR")
        
        # Add Article 1 under CRD
        tree.add_citation("CRD IV", CitationType.DIRECTIVE)
        tree.add_citation("Article 1", CitationType.ARTICLE, parent_reference="CRD IV")
        
        # Should find both
        articles = tree.find_citations_by_reference("Article 1")
        assert len(articles) == 2
        
        # Contexts should be different
        contexts = {a.context for a in articles}
        assert "CRR > Article 1" in contexts
        assert "CRD IV > Article 1" in contexts
    
    def test_special_characters_in_reference(self):
        """Test references with special characters."""
        tree = RegulationCitationTree(name="special")
        
        tree.add_citation(
            "Article 92(1)(a)",
            CitationType.ARTICLE,
            text="Test with special chars"
        )
        
        node = tree.get_citation_by_reference("Article 92(1)(a)")
        assert node is not None
        assert node.reference == "Article 92(1)(a)"
    
    def test_large_text_content(self):
        """Test handling large text content."""
        tree = RegulationCitationTree(name="large_text")
        
        large_text = "Lorem ipsum " * 1000
        tree.add_citation(
            "Article 1",
            CitationType.ARTICLE,
            text=large_text
        )
        
        node = tree.get_citation_by_reference("Article 1")
        assert len(node.text) > 10000
        
        stats = tree.get_statistics()
        assert stats['total_text_length'] > 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
