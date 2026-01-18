# Their multi-stage approach:

STAGE 1: Document Segmentation
- Split legal documents by sections (similar to regulatory articles)
- Each section = potential training chunk
- Maintained hierarchical structure (Title > Chapter > Section > Paragraph)

STAGE 2: Synthetic Q&A Generation
# They used templates + extraction:

templates = [
    "What does [STATUTE_NAME] say about [CONCEPT]?",
    "Under [STATUTE_NAME], what are the requirements for [ACTION]?",
    "How is [TERM] defined in [STATUTE_NAME]?",
    "What are the penalties under [STATUTE_NAME] for [VIOLATION]?"
]

# Then extracted answers directly from statute text
# Example:
{
  "question": "What does Section 230 say about immunity?",
  "answer": "Section 230 provides that [EXACT QUOTE FROM STATUTE]...",
  "citation": "47 U.S.C. § 230(c)(1)",
  "source_text": "[full section text]"
}