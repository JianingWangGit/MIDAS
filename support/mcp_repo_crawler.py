# Repo/mcp_repo_crawler.py
#!/usr/bin/env python3
"""
MCP Client Repository Crawler
Identifies GitHub repositories that are MCP clients/hosts
(applications that embed MCP clients and connect to MCP servers at runtime)
"""

import os
import time
import json
import base64
import logging
import csv
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from dotenv import load_dotenv
import platform
import sys
import hashlib

from crawler_config import (
    SEARCH_QUERIES,
    MIN_STARS,
    MAX_REPOS_PER_QUERY,
    EXCLUDE_PATTERNS,
    KNOWN_CLIENTS,
    EVIDENCE_WEIGHTS,
    RATE_LIMIT_DELAY,
    MAX_WORKERS,
    OUTPUT_PREFIX,
    OUTPUT_FORMATS,
    CONFIDENCE_LEVELS,
    USE_AI_ASSIST,
)

# Load .env so GITHUB_TOKEN / OPENAI_API_KEY work in all entrypoints
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Repository:
    """Repository information and classification result."""
    full_name: str
    stars: int
    html_url: str
    language: str
    description: str
    topics: List[str]
    created_at: str
    updated_at: str
    default_branch: str
    confidence_score: float = 0.0
    classification: str = "unknown"
    evidence: List[str] = field(default_factory=list)
    ai_vote: Optional[str] = None  # optional AI-assisted label

class MCPRepoCrawler:
    """GitHub crawler for identifying MCP client applications."""

    def __init__(self, github_token: str, openai_api_key: Optional[str] = None):
        if not github_token:
            raise ValueError("GITHUB_TOKEN is required")

        self.github_token = github_token
        self.openai_api_key = openai_api_key
        self.use_ai_assist = USE_AI_ASSIST and bool(openai_api_key)

        self.base_url = "https://api.github.com"
        self.session = requests.Session()
        # token vs Bearer both work; this keeps things simple/compatible
        self.session.headers.update(
            {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
        )

        # Simple in-memory caches
        self.file_content_cache: Dict[str, Optional[str]] = {}
        self.repo_cache: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # GitHub API helpers
    # ------------------------------------------------------------------

    def search_repositories(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search GitHub repositories with pagination."""
        repos: List[Dict] = []
        page = 1
        per_page = 100

        while len(repos) < max_results:
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": min(per_page, max_results - len(repos)),
                "page": page,
            }
            url = f"{self.base_url}/search/repositories"
            try:
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                items = data.get("items", [])
                if not items:
                    break

                repos.extend(items)
                logger.info(
                    "Fetched %d repos (total: %d/%d) for query=%r",
                    len(items),
                    len(repos),
                    max_results,
                    query,
                )

                if len(repos) >= max_results or len(items) < per_page:
                    break

                page += 1
                time.sleep(RATE_LIMIT_DELAY)
            except Exception as e:
                logger.error("Error searching repositories [%s]: %s", query, e)
                break

        return repos[:max_results]

    def get_file_content(self, repo_full_name: str, path: str) -> Optional[str]:
        """Get raw text content of a file from a repository using /contents API."""
        cache_key = f"{repo_full_name}:{path}"
        if cache_key in self.file_content_cache:
            return self.file_content_cache[cache_key]

        url = f"{self.base_url}/repos/{repo_full_name}/contents/{path}"
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "content" in data:
                    content = base64.b64decode(data["content"]).decode(
                        "utf-8", errors="ignore"
                    )
                    self.file_content_cache[cache_key] = content
                    return content
        except Exception as e:
            logger.debug(
                "Could not fetch %s from %s: %s", path, repo_full_name, e
            )

        self.file_content_cache[cache_key] = None
        return None

    def search_code_in_repo(
        self, repo_full_name: str, term: str, max_results: int = 2
    ) -> List[Dict]:
        """
        Search for code within a specific repository.
        NOTE: code search is heavily rate-limited; we call this sparingly.
        """
        url = f"{self.base_url}/search/code"
        params = {
            "q": f"{term} repo:{repo_full_name}",
            "per_page": max_results,
        }

        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("items", [])
            elif resp.status_code == 422:
                # Some repos/patterns yield 422 from code search; just skip
                logger.debug("422 from code search for %s term=%r", repo_full_name, term)
                return []
            else:
                logger.debug(
                    "Code search failed for %s term=%r status=%s body=%s",
                    repo_full_name,
                    term,
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as e:
            logger.debug("Code search error for %s term=%r: %s", repo_full_name, term, e)

        return []

    # ------------------------------------------------------------------
    # Dependency / config heuristics
    # ------------------------------------------------------------------

    def check_npm_dependencies(self, content: str) -> Tuple[float, str]:
        """Check package.json for MCP client dependencies."""
        try:
            data = json.loads(content)
            deps = {
                **data.get("dependencies", {}),
                **data.get("devDependencies", {}),
            }

            if "@modelcontextprotocol/client" in deps:
                return EVIDENCE_WEIGHTS["package_dependency"], "Uses @modelcontextprotocol/client"
            if "@modelcontextprotocol/sdk" in deps:
                return EVIDENCE_WEIGHTS["package_dependency"] - 5, "Uses MCP SDK (may include client)"
            if "mcp-client" in deps:
                return EVIDENCE_WEIGHTS["package_dependency"], "Uses mcp-client package"

            mcp_deps = [
                d for d in deps if "mcp" in d.lower() or "model-context-protocol" in d.lower()
            ]
            if mcp_deps:
                client_deps = [
                    d for d in mcp_deps if "server" not in d.lower() or "client" in d.lower()
                ]
                if client_deps:
                    return (
                        EVIDENCE_WEIGHTS["package_dependency"] - 10,
                        f"Has MCP client-like deps: {', '.join(client_deps[:3])}",
                    )
                return (
                    EVIDENCE_WEIGHTS["package_dependency"] - 15,
                    f"Has MCP-related deps: {', '.join(mcp_deps[:3])}",
                )
        except Exception:
            pass
        return 0.0, ""

    def check_python_requirements(self, content: str) -> Tuple[float, str]:
        """Check requirements.txt for MCP client packages."""
        lines = content.lower().splitlines()
        mcp_packages = []

        for line in lines:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            if "mcp-client" in line or "mcp_client" in line:
                return EVIDENCE_WEIGHTS["package_dependency"], "Uses mcp-client package"
            if "modelcontextprotocol" in line:
                return EVIDENCE_WEIGHTS["package_dependency"] - 5, "Uses modelcontextprotocol package"
            if "mcp" in line and "server" not in line:
                pkg = (
                    line.split("==")[0]
                    .split(">=")[0]
                    .split("<=")[0]
                    .split("~=")[0]
                    .strip()
                )
                mcp_packages.append(pkg)

        if mcp_packages:
            return (
                EVIDENCE_WEIGHTS["package_dependency"] - 15,
                f"Has MCP-related Python packages: {', '.join(mcp_packages[:3])}",
            )
        return 0.0, ""

    def check_pyproject_toml(self, content: str) -> Tuple[float, str]:
        """Check pyproject.toml for MCP client dependencies."""
        lower = content.lower()

        if (
            "[tool.poetry.dependencies]" in lower
            or "[project.dependencies]" in lower
        ):
            if "mcp-client" in lower or "mcp_client" in lower:
                return EVIDENCE_WEIGHTS["package_dependency"], "Uses mcp-client package"
            if "modelcontextprotocol" in lower:
                return EVIDENCE_WEIGHTS["package_dependency"] - 5, "Uses modelcontextprotocol package"
            if '"mcp"' in lower or "'mcp'" in lower:
                return EVIDENCE_WEIGHTS["package_dependency"] - 10, "Has generic MCP dependency"

        if "mcp" in lower:
            if "client" in lower:
                return (
                    EVIDENCE_WEIGHTS["package_dependency"] - 10,
                    "Has MCP client-related configuration",
                )
            return (
                EVIDENCE_WEIGHTS["package_dependency"] - 15,
                "Mentions MCP in pyproject configuration",
            )

        return 0.0, ""

    # Other ecosystems – keep them simple for now

    def check_cargo_toml(self, content: str) -> Tuple[float, str]:
        lower = content.lower()
        if "mcp-client" in lower or "mcp_client" in lower:
            return EVIDENCE_WEIGHTS["package_dependency"] - 5, "Uses mcp-client crate"
        if "modelcontextprotocol" in lower:
            return EVIDENCE_WEIGHTS["package_dependency"] - 15, "Uses modelcontextprotocol crate"
        return 0.0, ""

    def check_go_mod(self, content: str) -> Tuple[float, str]:
        lower = content.lower()
        if "mcp/client" in lower or "mcp-client" in lower:
            return EVIDENCE_WEIGHTS["package_dependency"] - 5, "Uses MCP client Go module"
        if "modelcontextprotocol" in lower:
            return EVIDENCE_WEIGHTS["package_dependency"] - 15, "Uses modelcontextprotocol module"
        return 0.0, ""

    def check_maven_pom(self, content: str) -> Tuple[float, str]:
        lower = content.lower()
        if "mcp-client" in lower:
            return EVIDENCE_WEIGHTS["package_dependency"] - 5, "Uses MCP client Maven dependency"
        if "modelcontextprotocol" in lower:
            return EVIDENCE_WEIGHTS["package_dependency"] - 15, "Uses modelcontextprotocol dependency"
        return 0.0, ""

    def check_gradle(self, content: str) -> Tuple[float, str]:
        lower = content.lower()
        if "mcp-client" in lower:
            return EVIDENCE_WEIGHTS["package_dependency"] - 5, "Uses MCP client Gradle dependency"
        if "modelcontextprotocol" in lower:
            return EVIDENCE_WEIGHTS["package_dependency"] - 15, "Uses modelcontextprotocol Gradle dependency"
        return 0.0, ""

    def check_composer(self, content: str) -> Tuple[float, str]:
        try:
            data = json.loads(content)
            deps = {**data.get("require", {}), **data.get("require-dev", {})}
            for name in deps:
                lname = name.lower()
                if "mcp" in lname and "client" in lname:
                    return EVIDENCE_WEIGHTS["package_dependency"] - 10, "Has MCP client PHP package"
                if "modelcontextprotocol" in lname:
                    return EVIDENCE_WEIGHTS["package_dependency"] - 15, "Uses modelcontextprotocol PHP package"
        except Exception:
            pass
        return 0.0, ""

    # ------------------------------------------------------------------
    # Core repository analysis
    # ------------------------------------------------------------------

    def analyze_repository(self, repo_data: Dict) -> Repository:
        """Analyze a repository to determine if it's an MCP client application."""
        repo = Repository(
            full_name=repo_data["full_name"],
            stars=repo_data.get("stargazers_count", 0),
            html_url=repo_data["html_url"],
            language=repo_data.get("language") or "",
            description=repo_data.get("description") or "",
            topics=repo_data.get("topics", []),
            created_at=repo_data.get("created_at") or "",
            updated_at=repo_data.get("updated_at") or "",
            default_branch=repo_data.get("default_branch", "main"),
        )

        confidence = 0.0
        evidence: List[str] = []

        # 1) Known MCP hosts seed list
        if repo.full_name in KNOWN_CLIENTS:
            confidence += 50.0
            evidence.append("Known MCP client host (seed list)")

        desc_lower = repo.description.lower()
        topics_lower = " ".join(repo.topics).lower()

        # 2) Description & topics
        if "model context protocol" in desc_lower:
            confidence += EVIDENCE_WEIGHTS["description_mention"]
            evidence.append("Description explicitly mentions Model Context Protocol")
        elif "mcp" in desc_lower and any(
            t in desc_lower
            for t in ["client", "host", "integration", "support", "agent", "tool"]
        ):
            confidence += EVIDENCE_WEIGHTS["description_mention"] - 5
            evidence.append("Description mentions MCP with integration terms")

        if any(t in topics_lower for t in ["mcp", "model-context-protocol", "mcp-client", "mcp-host"]):
            confidence += EVIDENCE_WEIGHTS["topic_mention"]
            evidence.append("Repository topics include MCP-related tags")

        # 3) Application type indicators (agents, IDEs, bots, etc.)
        app_types = {
            "agent": 10,
            "ide": 10,
            "extension": 10,
            "plugin": 10,
            "cli": 8,
            "desktop": 8,
            "terminal": 8,
            "workflow": 10,
            "automation": 10,
            "chatbot": 8,
            "assistant": 8,
            "copilot": 10,
            "workspace": 8,
            "coding": 8,
            "development": 5,
        }
        for term, score in app_types.items():
            if term in desc_lower:
                confidence += score
                evidence.append(f"Application type suggests host: {term}")
                break

        # 4) Package files
        package_files = {
            "package.json": self.check_npm_dependencies,
            "requirements.txt": self.check_python_requirements,
            "pyproject.toml": self.check_pyproject_toml,
            "Cargo.toml": self.check_cargo_toml,
            "go.mod": self.check_go_mod,
            "pom.xml": self.check_maven_pom,
            "build.gradle": self.check_gradle,
            "composer.json": self.check_composer,
        }

        for filename, checker in package_files.items():
            content = self.get_file_content(repo.full_name, filename)
            if content:
                score, reason = checker(content)
                if score > 0:
                    confidence += score
                    evidence.append(f"{filename}: {reason}")

        # 5) MCP configuration files
        config_files = [
            ".mcp/config.json",
            "mcp.json",
            ".mcp.json",
            "mcp.config.json",
            "claude_desktop_config.json",
            ".continue/config.json",
        ]
        for cfg in config_files:
            content = self.get_file_content(repo.full_name, cfg)
            if not content:
                continue
            try:
                cfg_json = json.loads(content)
            except Exception:
                continue

            if any(k in cfg_json for k in ["servers", "mcpServers", "mcp", "tools"]):
                confidence += EVIDENCE_WEIGHTS["config_file"]
                evidence.append(f"Has MCP client configuration file: {cfg}")
                break

        # 6) README-based evidence
        readme_bonus = 0.0
        for readme_name in ["README.md", "readme.md", "README.rst", "README.txt"]:
            readme = self.get_file_content(repo.full_name, readme_name)
            if not readme:
                continue
            rl = readme.lower()

            if "model context protocol" in rl:
                readme_bonus = max(
                    readme_bonus, EVIDENCE_WEIGHTS["readme_mention"]
                )
                evidence.append("README explicitly documents Model Context Protocol usage")
            elif "mcp" in rl and any(
                p in rl
                for p in [
                    "mcp client",
                    "mcp host",
                    "mcp integration",
                    "mcp support",
                    "mcp servers",
                    "mcp tools",
                    "connect to mcp",
                ]
            ):
                readme_bonus = max(
                    readme_bonus,
                    EVIDENCE_WEIGHTS["readme_mention"] - 5,
                )
                evidence.append("README documents MCP client/host integration")
            elif "mcp" in rl and "server" in rl and any(
                p in rl for p in ["connect", "use", "integrate", "load"]
            ):
                readme_bonus = max(
                    readme_bonus,
                    EVIDENCE_WEIGHTS["readme_mention"] - 8,
                )
                evidence.append("README mentions using MCP servers from this app")

            break  # only inspect the first README we find

        confidence += readme_bonus

        # 7) Lightweight code search (ONLY if already somewhat MCP-ish)
        # Prevent API abuse by requiring a minimum confidence before code search.
        if confidence >= 15:
            client_patterns = [
                ("@modelcontextprotocol/client", EVIDENCE_WEIGHTS["client_code_pattern"], "Uses @modelcontextprotocol/client in code"),
                ("mcp.client", EVIDENCE_WEIGHTS["client_code_pattern"], "Uses mcp.client module in code"),
                ("MCPClient", EVIDENCE_WEIGHTS["client_code_pattern"], "Has MCPClient type/class in code"),
                ("StdioClientTransport", EVIDENCE_WEIGHTS["client_code_pattern"] - 5, "Uses StdioClientTransport (typical MCP client transport)"),
                ("connectToMCP", EVIDENCE_WEIGHTS["client_code_pattern"] - 5, "Has connectToMCP utility/function"),
            ]

            code_searches = 0
            CODE_SEARCH_LIMIT = 5

            for pattern, score, reason in client_patterns:
                if code_searches >= CODE_SEARCH_LIMIT:
                    break

                results = self.search_code_in_repo(repo.full_name, pattern, max_results=2)
                code_searches += 1

                if not results:
                    continue

                confidence += score
                evidence.append(reason)

                first_file = results[0].get("path", "").lower()
                if any(k in first_file for k in ["main", "index", "client", "app", "__init__"]):
                    confidence += 5
                    evidence.append(f"Core implementation file contains MCP client pattern: {first_file}")

        # 8) Simple recency boost
        try:
            if repo.updated_at:
                updated = datetime.fromisoformat(repo.updated_at.replace("Z", "+00:00"))
                if updated.year >= 2024:
                    confidence += 5
                    evidence.append("Recently updated (2024+), consistent with MCP ecosystem timeline")
        except Exception:
            pass

        # 9) Negative indicators / false positives
        if any(
            term in desc_lower
            for term in ["mcp server only", "mcp sdk", "sdk for model context protocol"]
        ):
            confidence -= 30
            evidence.append("Description suggests MCP server/SDK rather than client host")

        name_lower = repo.full_name.lower()
        for pat in ["awesome-mcp", "mcp-servers", "mcp-server-"]:
            if pat in name_lower:
                confidence -= 40
                evidence.append("Known non-client collection/template pattern")
                break

        # Clamp score and classify
        repo.confidence_score = max(0.0, min(100.0, confidence))
        repo.evidence = evidence

        if repo.confidence_score >= CONFIDENCE_LEVELS["high_confidence"]:
            repo.classification = "high_confidence_client"
        elif repo.confidence_score >= CONFIDENCE_LEVELS["likely"]:
            repo.classification = "likely_client"
        elif repo.confidence_score >= CONFIDENCE_LEVELS["possible"]:
            repo.classification = "possible_client"
        else:
            repo.classification = "unlikely_client"

        return repo

    # ------------------------------------------------------------------
    # Optional: AI-assisted classification (hybrid mode)
    # ------------------------------------------------------------------
    def use_openai_classification(self, repo: Repository) -> Optional[Tuple[str, float]]:
        """Use OpenAI API to semantically classify repo as MCP CLIENT / SERVER / SDK / UNKNOWN."""
        if not self.openai_api_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)

            prompt = f"""
            Analyze this GitHub repository and determine if it's an MCP (Model Context Protocol) client application.

            MCP clients:
            - embed MCP client functionality
            - connect to MCP servers
            - use MCP tools/resources at runtime
            - are applications (CLI, IDE plugin, desktop client, agent, etc.)

            NOT clients:
            - MCP server implementations
            - SDKs or libraries
            - infrastructure/tooling

            Repository: {repo.full_name}
            Description: {repo.description}
            Language: {repo.language}
            Evidence: {', '.join(repo.evidence[:5])}

            Respond with ONLY ONE WORD:
            CLIENT, SERVER, SDK, UNKNOWN
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )

            msg = response.choices[0].message
            # In the current SDK for text-only chat, content is a string
            raw = (msg.content or "").strip()
            classification = raw.upper()

            # Decide score effects
            if classification == "CLIENT":
                repo.ai_vote = "CLIENT"
                return "ai_confirmed_client", 15

            if classification in ("SERVER", "SDK"):
                repo.ai_vote = classification
                return "ai_rejected_not_client", -40

            repo.ai_vote = classification
            return None

        except Exception as e:
            print("AI classification failed:", e)
            return None


    # ------------------------------------------------------------------
    # Crawl orchestration + saving
    # ------------------------------------------------------------------

    def crawl(
        self,
        search_queries: List[str],
        max_repos_per_query: int,
        min_stars: int,
        exclude_repos: Optional[List[str]] = None,
    ) -> List[Repository]:
        """Main crawling function."""
        if exclude_repos is None:
            exclude_repos = []

        all_repos: List[Dict] = []
        seen: set[str] = set()

        for query in search_queries:
            logger.info("Searching query: %s", query)
            repos = self.search_repositories(query, max_repos_per_query)

            for rd in repos:
                name = rd["full_name"]
                if name in seen:
                    continue

                if rd.get("stargazers_count", 0) < min_stars:
                    continue

                lname = name.lower()
                if any(pat in lname for pat in EXCLUDE_PATTERNS):
                    logger.debug("Skipping excluded pattern repo: %s", name)
                    continue

                if name in exclude_repos:
                    logger.debug("Skipping explicitly excluded repo: %s", name)
                    continue

                seen.add(name)
                all_repos.append(rd)

        logger.info("Collected %d unique candidate repos", len(all_repos))

        analyzed: List[Repository] = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_name = {
                executor.submit(self.analyze_repository, rd): rd["full_name"]
                for rd in all_repos
            }

            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    repo = fut.result()

                    # Optional AI assist
                    if self.use_ai_assist and repo.confidence_score >= CONFIDENCE_LEVELS["possible"]:
                        ai_result = self.use_openai_classification(repo)
                        if ai_result:
                            reason, delta = ai_result
                            repo.confidence_score = max(
                                0.0, min(100.0, repo.confidence_score + delta)
                            )
                            repo.evidence.append(reason)
                            # Recompute classification after AI adjustment
                            if repo.confidence_score >= CONFIDENCE_LEVELS["high_confidence"]:
                                repo.classification = "high_confidence_client"
                            elif repo.confidence_score >= CONFIDENCE_LEVELS["likely"]:
                                repo.classification = "likely_client"
                            elif repo.confidence_score >= CONFIDENCE_LEVELS["possible"]:
                                repo.classification = "possible_client"
                            else:
                                repo.classification = "unlikely_client"

                    analyzed.append(repo)

                    if repo.classification in ("high_confidence_client", "likely_client"):
                        logger.info(
                            "Detected MCP client host: %s (confidence=%.1f)",
                            repo.full_name,
                            repo.confidence_score,
                        )
                except Exception as e:
                    logger.error("Error analyzing repo %s: %s", name, e)

        analyzed.sort(key=lambda r: r.confidence_score, reverse=True)
        return analyzed

    def save_results(self, repos: List[Repository], csv_file: str) -> None:
        """Save results to CSV."""
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "full_name",
                "stars",
                "html_url",
                "language",
                "description",
                "classification",
                "confidence_score",
                "topics",
                "evidence",
                "ai_vote",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in repos:
                writer.writerow(
                    {
                        "full_name": r.full_name,
                        "stars": r.stars,
                        "html_url": r.html_url,
                        "language": r.language,
                        "description": (r.description or "")[:200],
                        "classification": r.classification,
                        "confidence_score": r.confidence_score,
                        "topics": ", ".join(r.topics or []),
                        "evidence": " | ".join(r.evidence or []),
                        "ai_vote": r.ai_vote or "",
                    }
                )
        logger.info("CSV results saved to %s", csv_file)

    def save_json_results(self, repos: List[Repository], json_file: str) -> None:
        """Save detailed results to JSON."""
        data = [asdict(r) for r in repos]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("JSON results saved to %s", json_file)

    def save_metadata(self, metadata_file: str) -> None:
        """
        Save crawl process metadata for transparency/reproducibility.

        NOTE: This intentionally does NOT store per-repo counts or dataset-level
        statistics, only environment + configuration + model settings.
        """
        # Infer token type from prefix (no actual token is written)
        token_type = "unknown"
        if self.github_token.startswith("github_pat_"):
            token_type = "fine_grained_pat"
        elif self.github_token.startswith("ghp_"):
            token_type = "classic_pat"

        # Hash of the search queries for reproducibility
        queries_str = "\n".join(SEARCH_QUERIES)
        queries_hash = hashlib.sha256(queries_str.encode("utf-8")).hexdigest()

        metadata = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "python_version": sys.version.split()[0],
            "os": platform.platform(),
            "github_token_type": token_type,
            "openai_assist_enabled": self.use_ai_assist,
            "openai_model": "gpt-4o-mini" if self.use_ai_assist else None,
            "search_query_count": len(SEARCH_QUERIES),
            "known_clients_count": len(KNOWN_CLIENTS),
            "search_queries_sha256": queries_hash,
            "crawler_params": {
                "min_stars": MIN_STARS,
                "max_repos_per_query": MAX_REPOS_PER_QUERY,
                "rate_limit_delay": RATE_LIMIT_DELAY,
                "max_workers": MAX_WORKERS,
                "confidence_levels": CONFIDENCE_LEVELS,
                "use_ai_assist_config_flag": USE_AI_ASSIST,
            },
        }

        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info("Metadata saved to %s", metadata_file)
        except Exception as e:
            logger.error("Failed to write metadata file %s: %s", metadata_file, e)



def main():
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        logger.error("GITHUB_TOKEN is not set. Put it in .env or environment.")
        raise SystemExit(1)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        logger.info("OPENAI_API_KEY found – AI-assisted classification is available.")
    else:
        logger.info("No OPENAI_API_KEY – running in pure heuristic mode.")

    crawler = MCPRepoCrawler(github_token=github_token, openai_api_key=openai_api_key)

    logger.info(
        "Starting crawl with %d queries (max %d repos/query, min stars=%d)",
        len(SEARCH_QUERIES),
        MAX_REPOS_PER_QUERY,
        MIN_STARS,
    )

    repos = crawler.crawl(
        search_queries=SEARCH_QUERIES,
        max_repos_per_query=MAX_REPOS_PER_QUERY,
        min_stars=MIN_STARS,
        exclude_repos=[],
    )

    high_conf = [r for r in repos if r.classification == "high_confidence_client"]
    likely = [r for r in repos if r.classification == "likely_client"]
    possible = [r for r in repos if r.classification == "possible_client"]

    logger.info("Total analyzed: %d", len(repos))
    logger.info("High confidence: %d", len(high_conf))
    logger.info("Likely clients: %d", len(likely))
    logger.info("Possible clients: %d", len(possible))

    # Save outputs
    base = OUTPUT_PREFIX or "mcp_clients"
    all_csv = f"{base}_all.csv"
    all_json = f"{base}_all.json"

    crawler.save_results(repos, all_csv)
    crawler.save_json_results(repos, all_json)

    high_quality = high_conf + likely
    if high_quality:
        crawler.save_results(high_quality, f"{base}_high_conf.csv")
        crawler.save_json_results(high_quality, f"{base}_high_conf.json")

    # Save crawl process metadata for transparency
    crawler.save_metadata("mcp_crew_process_metadata.json")

    # Quick console preview
    print("\n=== Top MCP Client Applications ===\n")
    for r in high_quality[:20]:
        print(f"{r.full_name} ⭐ {r.stars}")
        print(f"  URL: {r.html_url}")
        print(f"  Confidence: {r.confidence_score:.1f}% ({r.classification})")
        print(f"  Evidence: {'; '.join(r.evidence[:3])}")
        if r.ai_vote:
            print(f"  AI vote: {r.ai_vote}")
        print()


if __name__ == "__main__":
    main()
