# Their approach for sparse medical Q&A data:

1. SEED DATA (Manual):
   - 100-200 expert-written Q&A pairs from clinical guidelines
   - Covered key medical concepts with proper citations

2. SYNTHETIC EXPANSION:
   - Used GPT-4 to generate questions from:
     * Medical textbooks
     * Clinical practice guidelines  
     * Drug monographs
   - Generated 3-5 questions per page/section
   
3. EXPERT VALIDATION:
   - Medical doctors reviewed 20% of synthetic data
   - Acceptance rate: ~75% with minor edits
   
4. CONSISTENCY FILTERING:
   - Asked same question 3 times to GPT-4
   - Only kept Q&A pairs where answers were consistent
   - Removed contradictory or vague responses