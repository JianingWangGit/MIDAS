#!/usr/bin/env python3
"""
Utility script for analyzing and filtering MCP crawler results
"""

import json
import csv
import sys
from typing import List, Dict, Set
from collections import Counter

def load_results(filename: str) -> List[Dict]:
    """Load results from JSON or CSV file"""
    if filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    elif filename.endswith('.csv'):
        results = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['confidence_score'] = float(row.get('confidence_score', 0))
                row['stars'] = int(row.get('stars', 0))
                results.append(row)
        return results
    else:
        raise ValueError("Unsupported file format")

def analyze_evidence(results: List[Dict]):
    """Analyze evidence patterns across results"""
    print("\n=== Evidence Analysis ===\n")

    evidence_counter = Counter()

    for repo in results:
        if 'evidence' in repo:
            if isinstance(repo['evidence'], list):
                evidences = repo['evidence']
            else:
                evidences = repo['evidence'].split(' | ')

            for evidence in evidences:
                # Extract evidence type
                if 'package.json' in evidence:
                    evidence_counter['package.json dependency'] += 1
                elif 'requirements.txt' in evidence or 'pyproject.toml' in evidence:
                    evidence_counter['Python dependency'] += 1
                elif 'README' in evidence:
                    evidence_counter['README mention'] += 1
                elif 'Description' in evidence:
                    evidence_counter['Description mention'] += 1
                elif 'config file' in evidence:
                    evidence_counter['Config file'] += 1
                elif 'Client implementation' in evidence:
                    evidence_counter['Client code'] += 1
                else:
                    evidence_counter[evidence[:30]] += 1

    print("Most common evidence types:")
    for evidence, count in evidence_counter.most_common(10):
        print(f"  - {evidence}: {count}")

def analyze_languages(results: List[Dict]):
    """Analyze language distribution"""
    print("\n=== Language Distribution ===\n")

    languages = Counter()
    for repo in results:
        lang = repo.get('language', 'Unknown')
        if lang:
            languages[lang] += 1

    total = sum(languages.values())
    for lang, count in languages.most_common():
        percentage = (count / total) * 100
        print(f"  {lang}: {count} ({percentage:.1f}%)")

def filter_by_confidence(results: List[Dict], min_confidence: float) -> List[Dict]:
    """Filter results by minimum confidence score"""
    return [r for r in results if r.get('confidence_score', 0) >= min_confidence]

def filter_by_stars(results: List[Dict], min_stars: int) -> List[Dict]:
    """Filter results by minimum star count"""
    return [r for r in results if r.get('stars', 0) >= min_stars]

def find_duplicates(results: List[Dict]) -> Set[str]:
    """Find duplicate repositories"""
    seen = set()
    duplicates = set()

    for repo in results:
        name = repo.get('full_name')
        if name in seen:
            duplicates.add(name)
        seen.add(name)

    return duplicates

def export_filtered(results: List[Dict], output_file: str,
                   min_confidence: float = 25, min_stars: int = 50):
    """Export filtered results to new file"""
    filtered = filter_by_confidence(results, min_confidence)
    filtered = filter_by_stars(filtered, min_stars)

    # Sort by confidence, then stars
    filtered.sort(key=lambda x: (x.get('confidence_score', 0), x.get('stars', 0)), reverse=True)

    if output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(filtered, f, indent=2)
    elif output_file.endswith('.csv'):
        if filtered:
            keys = filtered[0].keys()
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(filtered)

    print(f"\nExported {len(filtered)} filtered results to {output_file}")

def compare_with_known(results: List[Dict], known_file: str):
    """Compare results with known MCP clients"""
    print("\n=== Comparison with Known Clients ===\n")

    # Load known clients
    with open(known_file, 'r') as f:
        if known_file.endswith('.csv'):
            reader = csv.DictReader(f)
            known = {row['full_name'] for row in reader}
        else:
            known_data = json.load(f)
            known = {item['full_name'] for item in known_data}

    found = {r['full_name'] for r in results}

    # Analysis
    overlap = found & known
    missing = known - found
    new_finds = found - known

    print(f"Known clients found: {len(overlap)}/{len(known)} ({len(overlap)/len(known)*100:.1f}%)")
    print(f"Known clients missing: {len(missing)}")
    print(f"New discoveries: {len(new_finds)}")

    if missing:
        print("\nMissing known clients:")
        for repo in sorted(missing)[:10]:
            print(f"  - {repo}")

    return overlap, missing, new_finds

def generate_report(results: List[Dict], output_file: str = 'analysis_report.md'):
    """Generate a markdown report of the analysis"""

    high_conf = filter_by_confidence(results, 50)
    likely = filter_by_confidence(results, 25)

    report = f"""# MCP Client Repository Analysis Report

## Summary Statistics

- **Total repositories analyzed**: {len(results)}
- **High confidence clients**: {len(high_conf)}
- **Likely clients (confidence ≥25)**: {len(likely)}
- **Average confidence score**: {sum(r.get('confidence_score', 0) for r in results) / len(results):.1f}
- **Average star count**: {sum(r.get('stars', 0) for r in results) / len(results):.0f}

## Top MCP Client Applications

| Repository | Stars | Language | Confidence | Evidence |
|------------|-------|----------|------------|----------|
"""

    for repo in high_conf[:20]:
        evidence = repo.get('evidence', '')
        if isinstance(evidence, list):
            evidence = ' | '.join(evidence[:2])
        else:
            evidence = evidence.split(' | ')[0] if evidence else ''

        report += f"| [{repo['full_name']}]({repo.get('html_url', '')}) | {repo.get('stars', 0):,} | {repo.get('language', '')} | {repo.get('confidence_score', 0):.1f}% | {evidence[:50]}... |\n"

    # Add language distribution
    languages = Counter(r.get('language', 'Unknown') for r in likely)
    report += "\n## Language Distribution (Likely Clients)\n\n"
    for lang, count in languages.most_common(10):
        report += f"- **{lang}**: {count} repositories\n"

    # Add classification distribution
    classifications = Counter(r.get('classification', 'unknown') for r in results)
    report += "\n## Classification Distribution\n\n"
    for classification, count in classifications.most_common():
        report += f"- **{classification}**: {count} repositories\n"

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\nReport generated: {output_file}")

def main():
    """Main utility function"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results.json> [command]")
        print("\nCommands:")
        print("  analyze    - Analyze evidence and language patterns")
        print("  filter     - Filter by confidence and stars")
        print("  report     - Generate markdown report")
        print("  compare    - Compare with known clients")
        return

    results_file = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else 'analyze'

    # Load results
    results = load_results(results_file)
    print(f"Loaded {len(results)} repositories from {results_file}")

    if command == 'analyze':
        analyze_evidence(results)
        analyze_languages(results)

        # Find duplicates
        duplicates = find_duplicates(results)
        if duplicates:
            print(f"\n⚠️  Found {len(duplicates)} duplicate entries")

    elif command == 'filter':
        min_conf = float(input("Minimum confidence score (0-100): ") or "25")
        min_stars = int(input("Minimum star count: ") or "50")
        output = input("Output file (e.g., filtered.json): ") or "filtered.json"
        export_filtered(results, output, min_conf, min_stars)

    elif command == 'report':
        generate_report(results)

    elif command == 'compare':
        known_file = input("Known clients file (CSV/JSON): ") or "mcp_hosts_sampleV0.csv"
        compare_with_known(results, known_file)

    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()