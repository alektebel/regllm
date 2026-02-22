"""
Regulation Citation Tree (RCT) - A hierarchical data structure for managing
and retrieving exact regulation citations by context.

This module provides a tree-based API that allows agents to:
- Add citations with hierarchical context (e.g., Regulation > Article > Paragraph)
- Search for citations by context, keywords, or reference
- Build exact citation references by construction
- Navigate the citation hierarchy

Example citation hierarchy:
    CRR (Capital Requirements Regulation)
    └── Article 92
        ├── Paragraph 1
        │   └── Point (a)
        └── Paragraph 2
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CitationType(Enum):
    """Types of regulation citations."""
    REGULATION = "regulation"
    DIRECTIVE = "directive"
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    SECTION = "section"
    ANNEX = "annex"
    CHAPTER = "chapter"
    POINT = "point"
    SUBPOINT = "subpoint"
    TABLE = "table"
    FOOTNOTE = "footnote"


@dataclass
class CitationNode:
    """
    A node in the Regulation Citation Tree.
    
    Attributes:
        id: Unique identifier for this node
        type: Type of citation (regulation, article, paragraph, etc.)
        reference: The actual citation reference (e.g., "Article 92", "CRR")
        text: Full text content of this citation
        metadata: Additional metadata (date, source, tags, etc.)
        parent_id: ID of parent node (None for root)
        children_ids: List of child node IDs
        context: Full context path (e.g., "CRR > Article 92 > Paragraph 1")
        created_at: Timestamp when node was created
    """
    id: str
    type: CitationType
    reference: str
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    context: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        data = asdict(self)
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CitationNode':
        """Create node from dictionary."""
        data = data.copy()
        data['type'] = CitationType(data['type'])
        return cls(**data)


class RegulationCitationTree:
    """
    A hierarchical tree structure for managing regulation citations.
    
    This class provides an API for:
    - Adding citations with hierarchical relationships
    - Searching citations by context, keywords, or reference
    - Building exact citation paths
    - Traversing the citation hierarchy
    - Persisting and loading the tree
    """
    
    def __init__(self, name: str = "default", persist_path: Optional[str] = None):
        """
        Initialize a new Regulation Citation Tree.
        
        Args:
            name: Name/identifier for this tree
            persist_path: Path to save/load the tree (optional)
        """
        self.name = name
        self.nodes: Dict[str, CitationNode] = {}
        self.root_ids: List[str] = []
        self.persist_path = Path(persist_path) if persist_path else None
        
        # Index for fast lookup
        self._reference_index: Dict[str, List[str]] = {}  # reference -> node_ids
        self._type_index: Dict[CitationType, List[str]] = {}  # type -> node_ids
        self._keyword_index: Dict[str, List[str]] = {}  # keyword -> node_ids
        
        logger.info(f"Initialized RegulationCitationTree: {name}")
    
    def add_citation(
        self,
        reference: str,
        citation_type: Union[CitationType, str],
        text: str = "",
        parent_reference: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None
    ) -> str:
        """
        Add a citation to the tree.
        
        Args:
            reference: Citation reference (e.g., "Article 92", "CRR")
            citation_type: Type of citation
            text: Full text content of the citation
            parent_reference: Reference of the parent citation (None for root)
            metadata: Additional metadata
            node_id: Custom node ID (auto-generated if None)
        
        Returns:
            The node ID of the created citation
        
        Example:
            >>> tree = RegulationCitationTree()
            >>> crr_id = tree.add_citation("CRR", CitationType.REGULATION, 
            ...                            text="Capital Requirements Regulation")
            >>> art92_id = tree.add_citation("Article 92", CitationType.ARTICLE,
            ...                              parent_reference="CRR")
        """
        # Convert string type to enum
        if isinstance(citation_type, str):
            citation_type = CitationType(citation_type.lower())
        
        # Generate node ID if not provided
        if node_id is None:
            node_id = self._generate_node_id(reference, citation_type)
        
        # Find parent node if parent_reference is provided
        parent_id = None
        context = reference
        
        if parent_reference:
            parent_nodes = self._reference_index.get(parent_reference, [])
            if not parent_nodes:
                raise ValueError(f"Parent reference '{parent_reference}' not found")
            parent_id = parent_nodes[0]  # Use first match
            parent_node = self.nodes[parent_id]
            context = f"{parent_node.context} > {reference}"
        
        # Create the node
        node = CitationNode(
            id=node_id,
            type=citation_type,
            reference=reference,
            text=text,
            metadata=metadata or {},
            parent_id=parent_id,
            context=context
        )
        
        # Add to tree
        self.nodes[node_id] = node
        
        # Update parent's children
        if parent_id:
            self.nodes[parent_id].children_ids.append(node_id)
        else:
            self.root_ids.append(node_id)
        
        # Update indices
        self._update_indices(node)
        
        logger.info(f"Added citation: {context} (id: {node_id})")
        return node_id
    
    def get_citation(self, node_id: str) -> Optional[CitationNode]:
        """Get a citation node by ID."""
        return self.nodes.get(node_id)
    
    def get_citation_by_reference(self, reference: str) -> Optional[CitationNode]:
        """
        Get a citation by its reference.
        
        Args:
            reference: Citation reference to search for
        
        Returns:
            First matching CitationNode or None
        """
        node_ids = self._reference_index.get(reference, [])
        return self.nodes.get(node_ids[0]) if node_ids else None
    
    def find_citations_by_reference(self, reference: str) -> List[CitationNode]:
        """
        Find all citations matching a reference.
        
        Args:
            reference: Citation reference to search for
        
        Returns:
            List of matching CitationNodes
        """
        node_ids = self._reference_index.get(reference, [])
        return [self.nodes[nid] for nid in node_ids]
    
    def find_citations_by_type(self, citation_type: Union[CitationType, str]) -> List[CitationNode]:
        """
        Find all citations of a specific type.
        
        Args:
            citation_type: Type to search for
        
        Returns:
            List of matching CitationNodes
        """
        if isinstance(citation_type, str):
            citation_type = CitationType(citation_type.lower())
        
        node_ids = self._type_index.get(citation_type, [])
        return [self.nodes[nid] for nid in node_ids]
    
    def find_citations_by_keyword(self, keyword: str, case_sensitive: bool = False) -> List[CitationNode]:
        """
        Find citations containing a keyword in their text or reference.
        
        Args:
            keyword: Keyword to search for
            case_sensitive: Whether search should be case-sensitive
        
        Returns:
            List of matching CitationNodes
        """
        if not case_sensitive:
            keyword = keyword.lower()
        
        matching_nodes = []
        for node in self.nodes.values():
            text_to_search = node.text if case_sensitive else node.text.lower()
            ref_to_search = node.reference if case_sensitive else node.reference.lower()
            
            if keyword in text_to_search or keyword in ref_to_search:
                matching_nodes.append(node)
        
        return matching_nodes
    
    def find_citations_by_context(self, context_pattern: str) -> List[CitationNode]:
        """
        Find citations by context path pattern.
        
        Args:
            context_pattern: Pattern to match in context (e.g., "CRR > Article")
        
        Returns:
            List of matching CitationNodes
        
        Example:
            >>> nodes = tree.find_citations_by_context("CRR > Article 92")
        """
        matching_nodes = []
        for node in self.nodes.values():
            if context_pattern in node.context:
                matching_nodes.append(node)
        
        return matching_nodes
    
    def get_full_citation_path(self, node_id: str) -> List[CitationNode]:
        """
        Get the full path from root to a citation node.
        
        Args:
            node_id: ID of the target node
        
        Returns:
            List of CitationNodes from root to target
        """
        path = []
        current_node = self.nodes.get(node_id)
        
        while current_node:
            path.insert(0, current_node)
            current_node = self.nodes.get(current_node.parent_id) if current_node.parent_id else None
        
        return path
    
    def get_citation_text_with_context(self, node_id: str, include_children: bool = False) -> str:
        """
        Get formatted citation text with full context.
        
        Args:
            node_id: ID of the citation node
            include_children: Whether to include child citations
        
        Returns:
            Formatted citation text with context
        """
        node = self.nodes.get(node_id)
        if not node:
            return ""
        
        result = f"[{node.context}]\n{node.text}"
        
        if include_children and node.children_ids:
            result += "\n\nSub-citations:"
            for child_id in node.children_ids:
                child = self.nodes[child_id]
                result += f"\n  - {child.reference}: {child.text[:100]}..."
        
        return result
    
    def get_children(self, node_id: str) -> List[CitationNode]:
        """Get all direct children of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        return [self.nodes[child_id] for child_id in node.children_ids]
    
    def get_parent(self, node_id: str) -> Optional[CitationNode]:
        """Get the parent node of a citation."""
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return None
        
        return self.nodes.get(node.parent_id)
    
    def get_siblings(self, node_id: str) -> List[CitationNode]:
        """Get all sibling nodes (same parent)."""
        parent = self.get_parent(node_id)
        if not parent:
            return []
        
        return [self.nodes[child_id] for child_id in parent.children_ids if child_id != node_id]
    
    def update_citation(self, node_id: str, **kwargs) -> bool:
        """
        Update a citation's attributes.
        
        Args:
            node_id: ID of the node to update
            **kwargs: Attributes to update (text, metadata, etc.)
        
        Returns:
            True if successful, False otherwise
        """
        node = self.nodes.get(node_id)
        if not node:
            return False
        
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
        
        # Re-index if reference changed
        if 'reference' in kwargs:
            self._rebuild_indices()
        
        logger.info(f"Updated citation: {node.context}")
        return True
    
    def delete_citation(self, node_id: str, delete_children: bool = False) -> bool:
        """
        Delete a citation from the tree.
        
        Args:
            node_id: ID of the node to delete
            delete_children: If True, delete all children recursively
        
        Returns:
            True if successful, False otherwise
        """
        node = self.nodes.get(node_id)
        if not node:
            return False
        
        # Handle children
        if delete_children:
            for child_id in node.children_ids[:]:  # Copy list to avoid modification during iteration
                self.delete_citation(child_id, delete_children=True)
        elif node.children_ids:
            raise ValueError(f"Cannot delete node with children. Use delete_children=True or delete children first.")
        
        # Remove from parent's children list
        if node.parent_id:
            parent = self.nodes[node.parent_id]
            parent.children_ids.remove(node_id)
        else:
            self.root_ids.remove(node_id)
        
        # Remove from tree
        del self.nodes[node_id]
        
        # Rebuild indices
        self._rebuild_indices()
        
        logger.info(f"Deleted citation: {node.context}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the citation tree."""
        stats = {
            'total_nodes': len(self.nodes),
            'root_nodes': len(self.root_ids),
            'nodes_by_type': {},
            'max_depth': 0,
            'total_text_length': sum(len(node.text) for node in self.nodes.values())
        }
        
        # Count by type
        for citation_type in CitationType:
            count = len(self._type_index.get(citation_type, []))
            if count > 0:
                stats['nodes_by_type'][citation_type.value] = count
        
        # Calculate max depth
        for root_id in self.root_ids:
            depth = self._calculate_depth(root_id)
            stats['max_depth'] = max(stats['max_depth'], depth)
        
        return stats
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the citation tree to a JSON file.
        
        Args:
            path: Path to save to (uses self.persist_path if None)
        """
        save_path = Path(path) if path else self.persist_path
        if not save_path:
            raise ValueError("No save path specified")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'name': self.name,
            'root_ids': self.root_ids,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved citation tree to {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load the citation tree from a JSON file.
        
        Args:
            path: Path to load from (uses self.persist_path if None)
        """
        load_path = Path(path) if path else self.persist_path
        if not load_path:
            raise ValueError("No load path specified")
        
        if not load_path.exists():
            raise FileNotFoundError(f"Citation tree file not found: {load_path}")
        
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.name = data['name']
        self.root_ids = data['root_ids']
        self.nodes = {node_id: CitationNode.from_dict(node_data) 
                     for node_id, node_data in data['nodes'].items()}
        
        # Rebuild indices
        self._rebuild_indices()
        
        logger.info(f"Loaded citation tree from {load_path} ({len(self.nodes)} nodes)")
    
    def export_markdown(self, output_path: str, max_depth: Optional[int] = None) -> None:
        """
        Export the citation tree as a Markdown document.
        
        Args:
            output_path: Path to save the Markdown file
            max_depth: Maximum depth to export (None for all)
        """
        lines = [f"# {self.name} - Citation Tree\n"]
        
        for root_id in self.root_ids:
            self._export_node_markdown(root_id, lines, depth=0, max_depth=max_depth)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported citation tree to {output_path}")
    
    def _export_node_markdown(self, node_id: str, lines: List[str], depth: int, max_depth: Optional[int]) -> None:
        """Recursively export node as Markdown."""
        if max_depth is not None and depth > max_depth:
            return
        
        node = self.nodes[node_id]
        indent = "  " * depth
        
        # Add node reference
        lines.append(f"{indent}- **{node.reference}** ({node.type.value})")
        
        # Add text if present
        if node.text:
            text_preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
            lines.append(f"{indent}  {text_preview}")
        
        # Add children
        for child_id in node.children_ids:
            self._export_node_markdown(child_id, lines, depth + 1, max_depth)
    
    def _generate_node_id(self, reference: str, citation_type: CitationType) -> str:
        """Generate a unique node ID."""
        base_id = f"{citation_type.value}_{reference.replace(' ', '_')}"
        node_id = base_id
        counter = 1
        
        while node_id in self.nodes:
            node_id = f"{base_id}_{counter}"
            counter += 1
        
        return node_id
    
    def _update_indices(self, node: CitationNode) -> None:
        """Update search indices for a node."""
        # Reference index
        if node.reference not in self._reference_index:
            self._reference_index[node.reference] = []
        self._reference_index[node.reference].append(node.id)
        
        # Type index
        if node.type not in self._type_index:
            self._type_index[node.type] = []
        self._type_index[node.type].append(node.id)
    
    def _rebuild_indices(self) -> None:
        """Rebuild all search indices."""
        self._reference_index.clear()
        self._type_index.clear()
        
        for node in self.nodes.values():
            self._update_indices(node)
    
    def _calculate_depth(self, node_id: str, current_depth: int = 0) -> int:
        """Calculate the maximum depth from a node."""
        node = self.nodes[node_id]
        if not node.children_ids:
            return current_depth
        
        max_child_depth = current_depth
        for child_id in node.children_ids:
            child_depth = self._calculate_depth(child_id, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def __repr__(self) -> str:
        return f"RegulationCitationTree(name='{self.name}', nodes={len(self.nodes)}, roots={len(self.root_ids)})"
    
    def __len__(self) -> int:
        return len(self.nodes)
