"""
Example usage of the Regulation Citation Tree (RCT).

This script demonstrates how to use the RCT data structure to build
and retrieve exact citation references.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.citation_tree import RegulationCitationTree, CitationType


def example_basic_usage():
    """Basic usage: creating a simple citation tree."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Create a new citation tree
    tree = RegulationCitationTree(name="EU Banking Regulations")
    
    # Add a root regulation
    crr_id = tree.add_citation(
        reference="CRR",
        citation_type=CitationType.REGULATION,
        text="Regulation (EU) No 575/2013 on prudential requirements for credit institutions",
        metadata={
            "full_name": "Capital Requirements Regulation",
            "year": 2013,
            "jurisdiction": "EU"
        }
    )
    print(f"✓ Added CRR regulation (id: {crr_id})")
    
    # Add an article under CRR
    art92_id = tree.add_citation(
        reference="Article 92",
        citation_type=CitationType.ARTICLE,
        text="Own funds requirements",
        parent_reference="CRR",
        metadata={"topic": "capital requirements"}
    )
    print(f"✓ Added Article 92 (id: {art92_id})")
    
    # Add paragraphs under Article 92
    para1_id = tree.add_citation(
        reference="Paragraph 1",
        citation_type=CitationType.PARAGRAPH,
        text="Point (a): CET1 capital ratio of 4.5% of total risk exposure",
        parent_reference="Article 92"
    )
    print(f"✓ Added Paragraph 1 (id: {para1_id})")
    
    # Get the full citation with context
    print("\nFull citation with context:")
    citation_text = tree.get_citation_text_with_context(para1_id)
    print(citation_text)
    
    print()


def example_searching():
    """Example: searching for citations."""
    print("=" * 60)
    print("EXAMPLE 2: Searching Citations")
    print("=" * 60)
    
    tree = RegulationCitationTree(name="Banking Regulations")
    
    # Build a regulation tree
    tree.add_citation("CRR", CitationType.REGULATION, 
                     text="Capital Requirements Regulation")
    tree.add_citation("Article 92", CitationType.ARTICLE,
                     text="Own funds requirements", parent_reference="CRR")
    tree.add_citation("Article 93", CitationType.ARTICLE,
                     text="CET1 capital requirements", parent_reference="CRR")
    
    tree.add_citation("CRD IV", CitationType.DIRECTIVE,
                     text="Capital Requirements Directive")
    tree.add_citation("Article 73", CitationType.ARTICLE,
                     text="Capital buffer requirements", parent_reference="CRD IV")
    
    # Search by reference
    print("1. Search by reference:")
    art92 = tree.get_citation_by_reference("Article 92")
    print(f"   Found: {art92.context}")
    
    # Search by type
    print("\n2. Search by type (all articles):")
    articles = tree.find_citations_by_type(CitationType.ARTICLE)
    for art in articles:
        print(f"   - {art.context}")
    
    # Search by keyword
    print("\n3. Search by keyword 'capital':")
    capital_refs = tree.find_citations_by_keyword("capital")
    for ref in capital_refs:
        print(f"   - {ref.reference}: {ref.text[:50]}...")
    
    # Search by context
    print("\n4. Search by context pattern 'CRR > Article':")
    crr_articles = tree.find_citations_by_context("CRR > Article")
    for art in crr_articles:
        print(f"   - {art.context}")
    
    print()


def example_navigation():
    """Example: navigating the citation hierarchy."""
    print("=" * 60)
    print("EXAMPLE 3: Navigation")
    print("=" * 60)
    
    tree = RegulationCitationTree(name="CRR Structure")
    
    # Build hierarchy
    tree.add_citation("CRR", CitationType.REGULATION)
    tree.add_citation("Article 92", CitationType.ARTICLE, parent_reference="CRR")
    tree.add_citation("Paragraph 1", CitationType.PARAGRAPH, parent_reference="Article 92")
    tree.add_citation("Paragraph 2", CitationType.PARAGRAPH, parent_reference="Article 92")
    tree.add_citation("Point (a)", CitationType.POINT, parent_reference="Paragraph 1")
    
    # Get a deep node
    point_a = tree.get_citation_by_reference("Point (a)")
    
    # Navigate up (get full path)
    print("1. Full path to Point (a):")
    path = tree.get_full_citation_path(point_a.id)
    for i, node in enumerate(path):
        indent = "  " * i
        print(f"{indent}→ {node.reference} ({node.type.value})")
    
    # Navigate to parent
    print("\n2. Parent of Point (a):")
    parent = tree.get_parent(point_a.id)
    print(f"   {parent.context}")
    
    # Navigate to children
    print("\n3. Children of Article 92:")
    art92 = tree.get_citation_by_reference("Article 92")
    children = tree.get_children(art92.id)
    for child in children:
        print(f"   - {child.reference}")
    
    # Get siblings
    print("\n4. Siblings of Paragraph 1:")
    para1 = tree.get_citation_by_reference("Paragraph 1")
    siblings = tree.get_siblings(para1.id)
    for sibling in siblings:
        print(f"   - {sibling.reference}")
    
    print()


def example_persistence():
    """Example: saving and loading the tree."""
    print("=" * 60)
    print("EXAMPLE 4: Persistence")
    print("=" * 60)
    
    # Create and populate tree
    tree = RegulationCitationTree(
        name="EU Regulations",
        persist_path="./data/citation_trees/eu_regulations.json"
    )
    
    tree.add_citation("CRR", CitationType.REGULATION,
                     text="Capital Requirements Regulation")
    tree.add_citation("Article 92", CitationType.ARTICLE,
                     text="Own funds requirements",
                     parent_reference="CRR")
    
    # Save to file
    tree.save()
    print("✓ Saved tree to ./data/citation_trees/eu_regulations.json")
    
    # Load into new tree
    new_tree = RegulationCitationTree(
        name="Loaded Tree",
        persist_path="./data/citation_trees/eu_regulations.json"
    )
    new_tree.load()
    print(f"✓ Loaded tree: {len(new_tree)} nodes")
    
    # Verify data
    art92 = new_tree.get_citation_by_reference("Article 92")
    print(f"✓ Verified: {art92.context}")
    
    print()


def example_agent_integration():
    """Example: using RCT in an agent workflow."""
    print("=" * 60)
    print("EXAMPLE 5: Agent Integration")
    print("=" * 60)
    
    # Initialize tree
    tree = RegulationCitationTree(name="Agent Knowledge Base")
    
    # Simulate agent building citations from regulatory documents
    print("Agent: Building citation tree from regulatory documents...\n")
    
    # Add Basel III framework
    tree.add_citation("Basel III", CitationType.REGULATION,
                     text="Basel III: International regulatory framework")
    tree.add_citation("Pillar 1", CitationType.CHAPTER,
                     text="Minimum capital requirements",
                     parent_reference="Basel III")
    tree.add_citation("Section 1.1", CitationType.SECTION,
                     text="Credit risk - standardized approach",
                     parent_reference="Pillar 1")
    
    # Agent query: Find all content related to credit risk
    print("User Query: 'Show me all citations about credit risk'")
    print("\nAgent: Searching citation tree...\n")
    
    results = tree.find_citations_by_keyword("credit risk")
    
    print(f"Agent: Found {len(results)} citation(s):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.context}")
        print(f"   Text: {result.text}")
        
        # Get full path for exact reference
        path = tree.get_full_citation_path(result.id)
        path_str = " → ".join([n.reference for n in path])
        print(f"   Exact Reference: {path_str}")
    
    # Get statistics
    print("\nAgent: Citation tree statistics:")
    stats = tree.get_statistics()
    for key, value in stats.items():
        if key != 'nodes_by_type':
            print(f"  {key}: {value}")
    
    print()


def example_complex_hierarchy():
    """Example: building a complex hierarchical structure."""
    print("=" * 60)
    print("EXAMPLE 6: Complex Hierarchy")
    print("=" * 60)
    
    tree = RegulationCitationTree(name="Complete CRR Article 92")
    
    # Build complete Article 92 structure
    tree.add_citation("CRR", CitationType.REGULATION,
                     text="Regulation (EU) No 575/2013")
    
    tree.add_citation("Article 92", CitationType.ARTICLE,
                     text="Own funds requirements",
                     parent_reference="CRR")
    
    # Paragraph 1
    tree.add_citation("Paragraph 1", CitationType.PARAGRAPH,
                     text="Institutions shall at all times satisfy the following own funds requirements:",
                     parent_reference="Article 92")
    
    tree.add_citation("Point (a)", CitationType.POINT,
                     text="CET1 capital ratio of 4.5%",
                     parent_reference="Paragraph 1")
    
    tree.add_citation("Point (b)", CitationType.POINT,
                     text="Tier 1 capital ratio of 6%",
                     parent_reference="Paragraph 1")
    
    tree.add_citation("Point (c)", CitationType.POINT,
                     text="Total capital ratio of 8%",
                     parent_reference="Paragraph 1")
    
    # Export structure
    print("Tree structure:")
    stats = tree.get_statistics()
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Nodes by type: {stats['nodes_by_type']}")
    
    # Export as markdown
    tree.export_markdown("./data/citation_trees/article_92_structure.md")
    print("\n✓ Exported to ./data/citation_trees/article_92_structure.md")
    
    # Show complete text with children
    art92 = tree.get_citation_by_reference("Article 92")
    print("\nComplete Article 92 with sub-citations:")
    print(tree.get_citation_text_with_context(art92.id, include_children=True))
    
    print()


def example_practical_use_case():
    """Example: practical use case for regulatory compliance."""
    print("=" * 60)
    print("EXAMPLE 7: Practical Use Case - Compliance Check")
    print("=" * 60)
    
    # Build citation tree for compliance checking
    tree = RegulationCitationTree(name="Capital Requirements Compliance")
    
    # Add relevant regulations
    tree.add_citation("CRR", CitationType.REGULATION,
                     text="Capital Requirements Regulation")
    
    tree.add_citation("Article 92", CitationType.ARTICLE,
                     text="Own funds requirements",
                     parent_reference="CRR",
                     metadata={"category": "capital", "mandatory": True})
    
    tree.add_citation("Paragraph 1(a)", CitationType.PARAGRAPH,
                     text="CET1 ratio of 4.5%",
                     parent_reference="Article 92",
                     metadata={
                         "threshold": 0.045,
                         "applies_to": "all institutions"
                     })
    
    tree.add_citation("Paragraph 1(b)", CitationType.PARAGRAPH,
                     text="Tier 1 capital ratio of 6%",
                     parent_reference="Article 92",
                     metadata={
                         "threshold": 0.06,
                         "applies_to": "all institutions"
                     })
    
    # Simulated compliance check
    print("Compliance Check Scenario:")
    print("Bank has CET1 ratio of 5.2%\n")
    
    # Find relevant requirement
    requirement = tree.get_citation_by_reference("Paragraph 1(a)")
    threshold = requirement.metadata.get("threshold")
    
    bank_ratio = 0.052
    
    print(f"Requirement: {requirement.context}")
    print(f"  Text: {requirement.text}")
    print(f"  Threshold: {threshold * 100}%")
    print(f"  Bank's ratio: {bank_ratio * 100}%")
    
    if bank_ratio >= threshold:
        print(f"  ✓ COMPLIANT (exceeds requirement by {(bank_ratio - threshold) * 100:.2f}%)")
    else:
        print(f"  ✗ NON-COMPLIANT (shortfall of {(threshold - bank_ratio) * 100:.2f}%)")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REGULATION CITATION TREE (RCT) - EXAMPLES")
    print("=" * 60 + "\n")
    
    example_basic_usage()
    example_searching()
    example_navigation()
    example_persistence()
    example_agent_integration()
    example_complex_hierarchy()
    example_practical_use_case()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
