# Regulation Citation Tree (RCT)

A hierarchical data structure for managing and retrieving exact regulation citations by context.

## Overview

The Regulation Citation Tree (RCT) provides a tree-based API that allows agents to:
- Add citations with hierarchical context (e.g., Regulation > Article > Paragraph)
- Search for citations by context, keywords, or reference
- Build exact citation references by construction
- Navigate the citation hierarchy
- Persist and load citation trees
- Export to Markdown format

## Features

- **Hierarchical Structure**: Organize citations in a tree structure reflecting regulatory document hierarchy
- **Multiple Citation Types**: Support for regulations, directives, articles, paragraphs, sections, points, and more
- **Fast Lookups**: Built-in indices for fast searching by reference, type, and keywords
- **Context Tracking**: Automatic context path building (e.g., "CRR > Article 92 > Paragraph 1")
- **Metadata Support**: Attach custom metadata to any citation node
- **Navigation**: Traverse the tree (parent, children, siblings, full path)
- **Persistence**: Save/load trees to/from JSON files
- **Export**: Export tree structure as Markdown

## Installation

The RCT is part of the RegLLM project. No additional dependencies are required beyond the base project requirements.

## Quick Start

```python
from src.citation_tree import RegulationCitationTree, CitationType

# Create a new tree
tree = RegulationCitationTree(name="EU Banking Regulations")

# Add a regulation
crr_id = tree.add_citation(
    reference="CRR",
    citation_type=CitationType.REGULATION,
    text="Capital Requirements Regulation (EU) No 575/2013"
)

# Add an article under the regulation
art92_id = tree.add_citation(
    reference="Article 92",
    citation_type=CitationType.ARTICLE,
    text="Own funds requirements",
    parent_reference="CRR"
)

# Add a paragraph under the article
para_id = tree.add_citation(
    reference="Paragraph 1",
    citation_type=CitationType.PARAGRAPH,
    text="Point (a): CET1 capital ratio of 4.5%",
    parent_reference="Article 92"
)

# Get the full citation with context
citation = tree.get_citation_text_with_context(para_id)
print(citation)
# Output:
# [CRR > Article 92 > Paragraph 1]
# Point (a): CET1 capital ratio of 4.5%
```

## Citation Types

The RCT supports the following citation types:

- `REGULATION` - EU regulations, national laws
- `DIRECTIVE` - EU directives
- `ARTICLE` - Articles within regulations/directives
- `PARAGRAPH` - Paragraphs within articles
- `SECTION` - Sections within documents
- `ANNEX` - Annexes to regulations
- `CHAPTER` - Chapters in regulatory frameworks
- `POINT` - Points within paragraphs (e.g., Point (a))
- `SUBPOINT` - Sub-points within points
- `TABLE` - Tables containing regulatory data
- `FOOTNOTE` - Footnotes and clarifications

## Core API

### Creating a Tree

```python
tree = RegulationCitationTree(
    name="My Citations",
    persist_path="./data/my_tree.json"  # Optional
)
```

### Adding Citations

```python
# Add root citation
node_id = tree.add_citation(
    reference="CRR",
    citation_type=CitationType.REGULATION,
    text="Full text of the regulation",
    metadata={"year": 2013, "jurisdiction": "EU"}
)

# Add child citation
child_id = tree.add_citation(
    reference="Article 92",
    citation_type=CitationType.ARTICLE,
    text="Article text",
    parent_reference="CRR",  # Links to parent
    metadata={"topic": "capital requirements"}
)
```

### Searching Citations

```python
# Search by reference
node = tree.get_citation_by_reference("Article 92")

# Search by type
articles = tree.find_citations_by_type(CitationType.ARTICLE)

# Search by keyword
results = tree.find_citations_by_keyword("capital")

# Search by context pattern
results = tree.find_citations_by_context("CRR > Article 92")
```

### Navigation

```python
# Get full path from root to node
path = tree.get_full_citation_path(node_id)

# Get parent
parent = tree.get_parent(node_id)

# Get children
children = tree.get_children(node_id)

# Get siblings
siblings = tree.get_siblings(node_id)
```

### Persistence

```python
# Save tree
tree.save("./data/my_tree.json")

# Load tree
tree.load("./data/my_tree.json")

# Export as Markdown
tree.export_markdown("./data/tree_structure.md")
```

### Update and Delete

```python
# Update citation
tree.update_citation(node_id, text="Updated text")

# Delete citation (must have no children)
tree.delete_citation(node_id)

# Delete citation and all children
tree.delete_citation(node_id, delete_children=True)
```

## Example Usage Scenarios

### 1. Agent Building a Knowledge Base

```python
# Initialize tree for agent
tree = RegulationCitationTree(name="Agent Knowledge Base")

# Agent processes regulatory documents
tree.add_citation("Basel III", CitationType.REGULATION)
tree.add_citation("Pillar 1", CitationType.CHAPTER, 
                 parent_reference="Basel III")

# Agent responds to query
results = tree.find_citations_by_keyword("credit risk")
for result in results:
    path = tree.get_full_citation_path(result.id)
    print(" → ".join([n.reference for n in path]))
```

### 2. Compliance Checking

```python
# Build requirements tree
tree = RegulationCitationTree(name="Compliance Requirements")

tree.add_citation(
    "Paragraph 1(a)",
    CitationType.PARAGRAPH,
    text="CET1 ratio of 4.5%",
    parent_reference="Article 92",
    metadata={"threshold": 0.045}
)

# Check compliance
requirement = tree.get_citation_by_reference("Paragraph 1(a)")
threshold = requirement.metadata["threshold"]
bank_ratio = 0.052

if bank_ratio >= threshold:
    print(f"✓ COMPLIANT: {requirement.context}")
```

### 3. Building Citation References

```python
# Get exact citation path
node = tree.get_citation_by_reference("Point (a)")
path = tree.get_full_citation_path(node.id)

# Build citation string
citation_str = " → ".join([n.reference for n in path])
print(f"Citation: {citation_str}")
# Output: Citation: CRR → Article 92 → Paragraph 1 → Point (a)
```

## Data Structure

Each citation node contains:

- `id`: Unique identifier
- `type`: Citation type (REGULATION, ARTICLE, etc.)
- `reference`: Citation reference (e.g., "Article 92")
- `text`: Full text content
- `metadata`: Custom metadata dictionary
- `parent_id`: ID of parent node
- `children_ids`: List of child node IDs
- `context`: Full context path (e.g., "CRR > Article 92")
- `created_at`: Timestamp

## Performance

The RCT uses indices for fast lookups:
- Reference index: O(1) lookup by reference
- Type index: O(1) lookup by citation type
- Context operations: O(depth) for path traversal

## Testing

Run the test suite:

```bash
python -m pytest tests/test_citation_tree.py -v
```

32 tests covering:
- Node creation and serialization
- Tree operations (add, update, delete)
- Searching and navigation
- Persistence and export
- Edge cases and error handling

## Examples

See `examples/citation_tree_examples.py` for comprehensive examples:

```bash
python examples/citation_tree_examples.py
```

Examples include:
1. Basic usage
2. Searching citations
3. Navigation
4. Persistence
5. Agent integration
6. Complex hierarchies
7. Practical compliance checking

## Integration with RegLLM

The RCT can be integrated with the RegLLM system to:

- Store and retrieve exact regulatory citations
- Build knowledge bases of regulatory structures
- Provide context-aware citation lookup for the RAG system
- Enable agents to construct precise regulatory references
- Track regulatory hierarchies for compliance checking

Example integration:

```python
from src.citation_tree import RegulationCitationTree
from src.rag_system import RegulatoryRAGSystem

# Initialize both systems
tree = RegulationCitationTree(name="Regulations")
rag = RegulatoryRAGSystem()

# Build citation tree from RAG documents
documents = rag.retrieve("CET1 requirements", n_results=10)
for doc in documents:
    # Extract citations and add to tree
    tree.add_citation(...)

# Use tree for exact citation lookup
citation = tree.get_citation_by_reference("Article 92")
```

## License

Part of the RegLLM project. See main project LICENSE.

## Contributing

Contributions welcome! Areas for improvement:
- Additional citation types
- Citation validation
- Cross-reference tracking
- Export formats (PDF, HTML)
- Integration with external citation databases
