# Repo/test_crawler.py
#!/usr/bin/env python3
"""
Test script for MCP Client Repository Crawler
Validates the crawler against known MCP clients
"""

import os
import sys
import json
import requests

from dotenv import load_dotenv

from mcp_repo_crawler import MCPRepoCrawler
from crawler_config import KNOWN_CLIENTS

load_dotenv()

# For deterministic tests: disable AI assist regardless of config
os.environ["USE_AI_ASSIST"] = "true"


def test_known_clients():
    """Test crawler against known MCP client repositories"""

    known_clients = KNOWN_CLIENTS

    github_token = os.environ.get('GITHUB_TOKEN')
    print("DEBUG GITHUB_TOKEN prefix:", repr((github_token or "")[:10]))
    if not github_token:
        print("Error: GITHUB_TOKEN not set")
        return False

    # 🔍 Quick token sanity check
    print("Checking GitHub token with /user ...")
    r = requests.get(
        "https://api.github.com/user",
        headers={
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
        },
        timeout=15,
    )
    print("  /user status:", r.status_code)

    if r.status_code == 401:
        print("  ❌ Token is invalid (401 Unauthorized).")
        return False
    elif r.status_code == 403 and "rate limit exceeded" in r.text.lower():
        print("  ⚠️ Token is valid but API rate limit is exceeded. Continuing tests anyway.")
    else:
        print("  ✅ Token appears valid.")

    print("\nTesting MCP Client Repository Crawler")
    print("=" * 50)

    # crawler = MCPRepoCrawler(github_token, openai_api_key=None)
    crawler = MCPRepoCrawler(
        github_token,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    results = {}

    for repo_name in known_clients:
        print(f"\nTesting: {repo_name}")

        # Get repository data
        url = f"https://api.github.com/repos/{repo_name}"
        response = crawler.session.get(url, timeout=30)

        if response.status_code != 200:
            print(f"  ❌ Failed to fetch repository: {response.status_code}")
            continue

        repo_data = response.json()

        # Analyze repository
        result = crawler.analyze_repository(repo_data)

        ev = (result.evidence or [])[:3]

        results[repo_name] = {
            'confidence': result.confidence_score,
            'classification': result.classification,
            'evidence': ev,  # Top 3 evidence
        }

        # Print results
        if result.confidence_score >= 50:
            status = "✅"
        elif result.confidence_score >= 25:
            status = "⚠️"
        elif result.confidence_score >= 10:
            status = "⚠️"
        else:
            status = "❌"

        print(f"  {status} Confidence: {result.confidence_score:.1f}%")
        print(f"  Classification: {result.classification}")
        print(f"  Evidence: {'; '.join(ev)}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if not results:
        print("⚠️ No repositories were successfully fetched/analyzed (likely all 401s).")
        return False

    high_confidence = sum(1 for r in results.values() if r['confidence'] >= 50)
    likely = sum(1 for r in results.values() if 25 <= r['confidence'] < 50)
    possible = sum(1 for r in results.values() if 10 <= r['confidence'] < 25)
    unlikely = sum(1 for r in results.values() if r['confidence'] < 10)

    print(f"High Confidence: {high_confidence}/{len(results)}")
    print(f"Likely: {likely}/{len(results)}")
    print(f"Possible: {possible}/{len(results)}")
    print(f"Unlikely: {unlikely}/{len(results)}")

    # Save test results
    with open('test_results_ai.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to test_results.json")

    detection_rate = (high_confidence + likely) / len(results)
    if detection_rate >= 0.7:
        print(f"\n✅ Test PASSED: {detection_rate:.0%} detection rate")
        return True
    else:
        print(f"\n❌ Test FAILED: {detection_rate:.0%} detection rate (expected >=70%)")
        return False


def test_search_queries():
    """Test that search queries are properly formatted"""
    from crawler_config import SEARCH_QUERIES

    print("\nTesting Search Queries")
    print("=" * 50)

    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        print("Error: GITHUB_TOKEN not set")
        return False

    crawler = MCPRepoCrawler(github_token, openai_api_key=None)

    # Test first 3 queries
    for query in SEARCH_QUERIES[:3]:
        print(f"\nQuery: {query}")
        repos = crawler.search_repositories(query, max_results=5)
        print(f"  Found {len(repos)} repositories")
        if repos:
            print(f"  Top result: {repos[0]['full_name']} ({repos[0]['stargazers_count']} stars)")

    return True


def main():
    """Run all tests"""
    print("MCP Client Repository Crawler Test Suite")
    print("=" * 70)

    # Test 1: Known clients
    test1_passed = test_known_clients()

    # Test 2: Search queries
    test2_passed = test_search_queries()

    print("\n" + "=" * 70)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
