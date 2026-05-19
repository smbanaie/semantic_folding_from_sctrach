#!/usr/bin/env python3
"""
Demo script showing the quick boundary-based splitting method and enhanced entity normalization.

This demonstrates how the new quick splitting method works for large textbooks
and documents without requiring resource-intensive embeddings, plus the enhanced
entity normalization that improves graph readability.
"""

import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.splitter_node import _quick_split_by_boundaries
from agents.reviewer_node import _normalize_entity


def demo_quick_splitting():
    """Demonstrate the quick boundary-based splitting method."""
    
    # Sample textbook-like content with clear chapter boundaries
    textbook_content = """
    INTRODUCTION TO COMPUTER SCIENCE
    
    Computer Science is the study of computers and computational systems.
    Unlike electrical and computer engineers, computer scientists deal mostly with software and software systems.
    This includes their theory, design, development, and application.
    
    PRINCIPAL AREAS OF STUDY
    
    Principal areas of study within Computer Science include artificial intelligence, computer systems and networks,
    security, database systems, human computer interaction, vision and graphics, numerical analysis, programming languages,
    software engineering, bioinformatics, and theory of computing.
    
    Although knowing how to program is essential to the study of computer science,
    it is only one element of the field. Computer scientists design and analyze algorithms
    to solve programs and study the performance of computer hardware and software.
    
    THE DISCIPLINE OF COMPUTER SCIENCE
    
    Computer Science is challenging because it requires both abstract thinking and practical skills.
    It involves understanding complex systems and creating solutions that are both efficient and effective.
    
    Many students are attracted to computer science because of the challenge and the potential for innovation.
    The field offers opportunities to work on cutting-edge technologies and solve real-world problems.
    
    CAREER OPPORTUNITIES
    
    Graduates with a degree in Computer Science can pursue careers in software development,
    data analysis, artificial intelligence, cybersecurity, and many other fields.
    
    The demand for computer science professionals continues to grow as technology becomes
    increasingly integrated into all aspects of modern life.
    """
    
    print("=== QUICK BOUNDARY-BASED SPLITTING DEMO ===\n")
    print(f"Original text length: {len(textbook_content)} characters")
    print(f"Original text lines: {len(textbook_content.splitlines())}")
    
    # Test with different numbers of target sections
    for target_sections in [2, 3, 4]:
        print(f"\n--- Splitting into {target_sections} sections ---")
        
        sections = _quick_split_by_boundaries(textbook_content, target_sections)
        
        print(f"Created {len(sections)} sections")
        
        for i, section in enumerate(sections):
            print(f"\nSection {i+1} ({len(section.content)} chars):")
            print(f"  Topic: {section.metadata.get('topic', 'Unknown')}")
            print(f"  Boundary type: {section.metadata.get('boundary_type', 'Unknown')}")
            print(f"  Content preview: {repr(section.content[:100])}...")
    
    # Verify content preservation
    sections = _quick_split_by_boundaries(textbook_content, 3)
    reconstructed = "".join(section.content for section in sections)
    
    print(f"\n=== CONTENT PRESERVATION CHECK ===")
    print(f"Original length: {len(textbook_content)}")
    print(f"Reconstructed length: {len(reconstructed)}")
    print(f"Content preserved: {reconstructed.strip() == textbook_content.strip()}")
    
    print(f"\n=== PERFORMANCE CHARACTERISTICS ===")
    print("✓ No resource-intensive embeddings required")
    print("✓ Fast processing for large documents")
    print("✓ Preserves original formatting and whitespace")
    print("✓ Uses natural boundaries (headers, paragraphs)")
    print("✓ Configurable via USE_QUICK_SPLITTING environment variable")


def demo_entity_normalization():
    """Demonstrate the enhanced entity normalization."""
    
    print("\n" + "="*60)
    print("=== ENHANCED ENTITY NORMALIZATION DEMO ===\n")
    
    # Examples of lengthy entities that are commonly found in knowledge graphs
    test_entities = [
        "Rich Relational Structure",
        "More Sophisticated Reasoning", 
        "Traditional RAG Approaches",
        "Large Language Models",
        "Questions That Require Connecting Information From Multiple Sources",
        "Opportunities For Explainable AI",
        "Explanations By Showing Graph Paths",
        "Several Challenges In Implementation And Deployment",
        "Computational Cost Of Building And Maintaining Knowledge Graphs",
        "Quality Of Extracted Knowledge",
        "Errors That Propagate Through The System",
        "Substantial Computational Resources",
        "Contextual Connections Between Concepts",
        "Hierarchical Structures",
        "Temporal Relationships",
        "Causal Relationships",
        "Semantic Associations"
    ]
    
    print("Entity Normalization Examples:")
    print("-" * 50)
    
    for entity in test_entities:
        normalized = _normalize_entity(entity)
        if entity != normalized:
            reduction = len(entity) - len(normalized)
            print(f"BEFORE: {entity}")
            print(f"AFTER:  {normalized}")
            print(f"REDUCTION: {reduction} characters ({(reduction/len(entity)*100):.1f}%)\n")
    
    # Calculate overall improvement
    original_total = sum(len(e) for e in test_entities)
    normalized_total = sum(len(_normalize_entity(e)) for e in test_entities)
    overall_reduction = original_total - normalized_total
    overall_percentage = (overall_reduction / original_total) * 100
    
    print(f"=== NORMALIZATION SUMMARY ===")
    print(f"Total characters before: {original_total}")
    print(f"Total characters after:  {normalized_total}")
    print(f"Total reduction:         {overall_reduction} characters")
    print(f"Overall improvement:     {overall_percentage:.1f}%")


if __name__ == "__main__":
    demo_quick_splitting()
    demo_entity_normalization()