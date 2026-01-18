# Multi-source hybrid approach:

DATA SOURCES:
1. Public legal databases (free access):
   - Court opinions (public domain)
   - Federal regulations (government websites)
   - Legal encyclopedias (some open access)

2. Synthetic generation from structure:
   - Identified common legal question patterns
   - Used GPT-4 to generate variations
   
3. Lawyer-in-the-loop:
   - Lawyers on staff reviewed ALL synthetic data
   - Correction rate: ~40% (high but acceptable)
   - Tracked common errors to improve generation prompts

4. Customer feedback loop:
   - Deployed early version to pilot customers
   - Collected real queries + lawyer corrections
   - Used this "production data" for fine-tuning