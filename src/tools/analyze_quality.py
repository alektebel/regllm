#!/usr/bin/env python3
"""
Analyze dataset quality and identify problematic samples.
"""

import json
import re
from pathlib import Path
import langdetect
from collections import defaultdict

def analyze_sample(sample, index):
    """Analyze a single sample for quality issues."""
    issues = []

    question = sample["messages"][1]["content"]
    answer = sample["messages"][2]["content"]
    metadata = sample.get("metadata", {})

    # Issue 1: Question asks about filename instead of concept
    # Check for actual filename patterns, not valid banking terms
    question_lower = question.lower()
    is_filename_question = False
    if ".pdf" in question_lower or "%20" in question_lower:
        is_filename_question = True
    elif "regulación sobre" in question_lower:
        # Only flag if it's about a filename/URL, not a banking concept
        banking_terms = ['minorista', 'capital', 'basilea', 'pd', 'lgd', 'ead', 'irb',
                        'crédito', 'riesgo', 'pyme', 'sme', 'provisión', 'mora',
                        'impago', 'solvencia', 'crd', 'crr']
        # Check if any banking term follows "regulación sobre"
        subject = question_lower.split("regulación sobre")[-1].strip().rstrip('?')
        if not any(term in subject for term in banking_terms):
            is_filename_question = True

    if is_filename_question:
        issues.append({
            "type": "filename_question",
            "severity": "high",
            "description": "Question asks about filename, not a conceptual question"
        })

    # Issue 2: Answer language
    try:
        # Check if answer is primarily in English
        answer_sample = answer[:500]  # First 500 chars
        lang = langdetect.detect(answer_sample)
        if lang != 'es':
            issues.append({
                "type": "wrong_language",
                "severity": "critical",
                "description": f"Answer in {lang}, not Spanish"
            })
    except:
        pass

    # Issue 3: Answer is just dumped text without structure
    if answer.count('\n') > 50 or len(answer) > 10000:
        # Very long answer with many line breaks = likely dumped text
        issues.append({
            "type": "unstructured_dump",
            "severity": "high",
            "description": "Answer appears to be unstructured text dump"
        })

    # Issue 4: Question is too vague or generic
    vague_patterns = [
        r"^¿Qué dice la regulación",
        r"^¿Qué metodología se usa para \w+\?$",
        r"^How is \w+ calculated\?$"
    ]
    for pattern in vague_patterns:
        if re.match(pattern, question, re.IGNORECASE):
            issues.append({
                "type": "vague_question",
                "severity": "medium",
                "description": "Question is too vague or generic"
            })
            break

    # Issue 5: Answer doesn't cite sources properly
    if not re.search(r'\[.*?\]', answer) and "Según" not in answer:
        issues.append({
            "type": "no_citation",
            "severity": "medium",
            "description": "Answer lacks proper source citations"
        })

    # Issue 6: Answer is irrelevant to question
    # Check for common keywords mismatch
    banking_keywords = ["capital", "pd", "lgd", "ead", "riesgo", "credit", "irb", "basilea", "basel", "crr", "tier"]
    question_lower = question.lower()
    answer_lower = answer.lower()

    # If question mentions banking terms, answer should too
    if any(kw in question_lower for kw in banking_keywords):
        if not any(kw in answer_lower for kw in banking_keywords):
            # Check if it's the transparency law (known mismatch)
            if "transparencia" in answer_lower and "buen gobierno" in answer_lower:
                issues.append({
                    "type": "irrelevant_answer",
                    "severity": "critical",
                    "description": "Answer about transparency law, not banking regulation"
                })

    # Issue 7: Question in English but should be in Spanish
    if question.startswith("How ") or question.startswith("What "):
        issues.append({
            "type": "english_question",
            "severity": "high",
            "description": "Question in English instead of Spanish"
        })

    # Issue 8: Very short answer (likely incomplete)
    if len(answer) < 100:
        issues.append({
            "type": "short_answer",
            "severity": "medium",
            "description": f"Answer too short ({len(answer)} chars)"
        })

    # Issue 9: Answer just says "No information" type response
    no_info_patterns = [
        "no tengo información",
        "no dispongo",
        "no está disponible",
        "no puedo responder"
    ]
    if any(pattern in answer_lower for pattern in no_info_patterns):
        issues.append({
            "type": "no_information",
            "severity": "high",
            "description": "Answer indicates no information available"
        })

    return {
        "index": index,
        "question": question[:150] + "..." if len(question) > 150 else question,
        "answer_length": len(answer),
        "source": metadata.get("source", "Unknown"),
        "issues": issues,
        "is_problematic": len(issues) > 0,
        "severity_score": sum(
            3 if i["severity"] == "critical" else
            2 if i["severity"] == "high" else 1
            for i in issues
        )
    }

def main():
    # Load dataset
    dataset_path = Path("data/processed/train_data.json")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📊 Analyzing {len(data)} samples...")
    print()

    # Analyze all samples
    results = []
    for i, sample in enumerate(data):
        result = analyze_sample(sample, i)
        results.append(result)

    # Statistics
    problematic = [r for r in results if r["is_problematic"]]
    critical = [r for r in results if any(i["severity"] == "critical" for i in r["issues"])]

    print("="*80)
    print("DATASET QUALITY ANALYSIS")
    print("="*80)
    print()
    print(f"Total samples: {len(data)}")
    print(f"Problematic samples: {len(problematic)} ({len(problematic)/len(data)*100:.1f}%)")
    print(f"Critical issues: {len(critical)} ({len(critical)/len(data)*100:.1f}%)")
    print()

    # Count issues by type
    issue_counts = defaultdict(int)
    for result in results:
        for issue in result["issues"]:
            issue_counts[issue["type"]] += 1

    print("Issues by type:")
    print("-" * 80)
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(data) * 100
        print(f"  {issue_type:30s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Samples to delete (critical + high severity issues)
    to_delete = [r for r in results if r["severity_score"] >= 3]

    print("="*80)
    print(f"RECOMMENDED FOR DELETION: {len(to_delete)} samples")
    print("="*80)
    print()

    # Show top 20 worst samples
    sorted_results = sorted(results, key=lambda x: x["severity_score"], reverse=True)

    print("Top 20 worst samples:")
    print("-" * 80)
    for result in sorted_results[:20]:
        print(f"\n[{result['index']}] Severity: {result['severity_score']}, Source: {result['source']}")
        print(f"Q: {result['question']}")
        for issue in result["issues"]:
            print(f"  ⚠️  {issue['severity'].upper()}: {issue['description']}")

    # Save report
    report = {
        "total_samples": len(data),
        "problematic_count": len(problematic),
        "critical_count": len(critical),
        "recommended_deletions": len(to_delete),
        "issue_counts": dict(issue_counts),
        "samples_to_delete": [r["index"] for r in to_delete],
        "detailed_results": sorted_results
    }

    with open("dataset_quality_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("="*80)
    print(f"✅ Report saved to: dataset_quality_report.json")
    print(f"✅ Indices to delete: {len(to_delete)} samples")
    print()
    print("Run clean_dataset.py to remove problematic samples")
    print("="*80)

if __name__ == "__main__":
    main()
