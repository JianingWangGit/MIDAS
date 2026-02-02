from github import Github, Auth, GithubException, RateLimitExceededException
import os
import csv
from itertools import islice
import re
import time
import random

from requests.exceptions import RetryError  # for "too many 403 error responses"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise SystemExit("GITHUB_TOKEN not found in environment")

PROCESSED_FILE = "NewData/processed_repos.txt"

# ---- CONFIG FLAGS ----

# For big runs, DISABLE comment scanning – GitHub secondary rate limit is very sensitive
# when you touch comments on every issue of a huge repo like lobehub/lobe-chat.
ENABLE_COMMENT_SCAN = False   # <- set to True only for small experiments


def load_processed_repos() -> set[str]:
    """Load repos already processed."""
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_processed_repo(repo_name: str):
    """Append a repo to the processed list."""
    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(repo_name + "\n")


# Use new auth style (no deprecation warning)
gh = Github(auth=Auth.Token(GITHUB_TOKEN))

INPUT_REPOS_CSV = "NewData/no_n_repos.csv"
# INPUT_REPOS_CSV = "Repo/mcp_hosts_sampleV0.csv"

# 分别输出到 open / closed 两个文件
OUTPUT_OPEN_ISSUES_CSV = "NewData/allopen.csv"
OUTPUT_CLOSED_ISSUES_CSV = "NewData/allclosed.csv"

# 更严格的 MCP 匹配规则：
# - 短语：model context protocol / modelcontextprotocol
# - "mcp" 必须是独立单词（避免 chatmcp 这种误伤）
MCP_TERMS = [
    "model context protocol",
    "modelcontextprotocol",
]

MCP_PATTERNS = [
    r"\bmcp\b",          # mcp 作为独立单词
    r"\bmcp server\b",
    r"\bmcp client\b",
    r"\bmcp tool\b",
    r"\bmcp protocol\b",
]


def contains_mcp(text: str) -> bool:
    """Check if text clearly refers to MCP (Model Context Protocol)."""
    text = (text or "").lower()

    # 先用简单子串匹配长短语
    if any(term in text for term in MCP_TERMS):
        return True

    # 再用正则匹配 "mcp" 独立单词及常见搭配
    for pat in MCP_PATTERNS:
        if re.search(pat, text):
            return True

    return False


def has_chinese(text: str) -> bool:
    """Return True if text contains any CJK Chinese characters."""
    text = text or ""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def looks_mcp_related(issue) -> bool:
    """
    Check title + full body + (optional) first 3 comments for MCP terms, skip Chinese.

    优化策略：
    1. 默认只看 title/body（不额外打 API）
    2. 如果 ENABLE_COMMENT_SCAN == True，才去拉前 3 条评论
    """
    global ENABLE_COMMENT_SCAN

    title = issue.title or ""
    body = issue.body or ""
    combined = f"{title}\n{body}"

    # 内容里有中文就直接跳过
    if has_chinese(combined):
        return False

    # 先在 title/body 中查 MCP
    if contains_mcp(combined):
        return True

    # 默认不再看评论，避免大量 403 / secondary rate limit
    if not ENABLE_COMMENT_SCAN:
        return False

    # 若显式启用了评论扫描，则最多拉前三条评论
    parts = [combined]
    try:
        for c in islice(issue.get_comments(), 3):
            parts.append(c.body or "")
    except RetryError as e:
        # 这是你看到的 "too many 403 error responses"
        print(f"  [RetryError fetching comments] {e}. "
              f"Disabling comment scanning for the rest of this run.")
        ENABLE_COMMENT_SCAN = False
        combined_all = combined
    except GithubException as e:
        # 常见是 status=403（secondary rate limit / abuse detection）
        print(f"  [GithubException fetching comments] status={getattr(e, 'status', '?')}, "
              f"data={getattr(e, 'data', '?')}. "
              f"Disabling comment scanning for the rest of this run.")
        ENABLE_COMMENT_SCAN = False
        combined_all = combined
    except Exception as e:
        # 其他异常：只打印，不再重试评论，以免刷屏
        print(f"  !! Failed to fetch comments for issue {getattr(issue, 'html_url', '?')}: {e}")
        combined_all = combined
    else:
        combined_all = "\n".join(parts)

    if has_chinese(combined_all):
        return False

    return contains_mcp(combined_all)


def summarize_issue(issue, max_body_chars: int = 200) -> str:
    """Body-only snippet, no title included."""
    body = (issue.body or "").replace("\n", " ").strip()
    if not body:
        return ""
    return body[:max_body_chars]


def load_target_repos(path: str) -> list[str]:
    repos = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full_name = (row.get("full_name") or "").strip()
            if full_name:
                repos.append(full_name)
    return repos


def main():
    target_repos = load_target_repos(INPUT_REPOS_CSV)
    processed_repos = load_processed_repos()

    print(f"Loaded {len(target_repos)} repos from CSV")
    print(f"Previously processed repos: {len(processed_repos)}")

    # 看看两个输出文件之前是否存在，用来决定要不要写 header
    open_exists = os.path.exists(OUTPUT_OPEN_ISSUES_CSV)
    closed_exists = os.path.exists(OUTPUT_CLOSED_ISSUES_CSV)

    with open(OUTPUT_OPEN_ISSUES_CSV, "a", newline="", encoding="utf-8") as out_open_f, \
         open(OUTPUT_CLOSED_ISSUES_CSV, "a", newline="", encoding="utf-8") as out_closed_f:

        fieldnames = ["repo", "issue_url", "title", "language", "description"]

        open_writer = csv.DictWriter(out_open_f, fieldnames=fieldnames)
        closed_writer = csv.DictWriter(out_closed_f, fieldnames=fieldnames)

        if not open_exists:
            open_writer.writeheader()
        if not closed_exists:
            closed_writer.writeheader()

        total_open = 0
        total_closed = 0

        for full_name in target_repos:
            if full_name in processed_repos:
                print(f"Skipping {full_name} (already processed)")
                continue

            print(f"\n=== Scanning issues in {full_name} ===")
            try:
                repo = gh.get_repo(full_name)
            except Exception as e:
                print(f"  !! Failed to get repo {full_name}: {e}")
                continue

            # repo language for all issues
            repo_lang = repo.language or ""

            try:
                # 仍然用 all，一次性拿到 open + closed
                issues = repo.get_issues(state="all")
            except Exception as e:
                print(f"  !! Failed to list issues for {full_name}: {e}")
                continue

            # 标记这个 repo 在遍历 issues 时是否出错（403 / rate limit 等）
            repo_had_error = False

            try:
                for issue in issues:
                    # 轻微降速，减少触发 secondary rate limit / 403 的概率
                    time.sleep(random.uniform(0.4, 1.3))

                    # Skip PRs. If this ever hits a fatal error, stop for this repo.
                    try:
                        if getattr(issue, "pull_request", None) is not None:
                            continue
                    except RetryError as e:
                        repo_had_error = True
                        print(f"  [RetryError while inspecting issue for PR] {e}. "
                              f"Stop early for {full_name}.")
                        break
                    except GithubException as e:
                        repo_had_error = True
                        print(f"  [GithubException while inspecting issue for PR] "
                              f"status={getattr(e, 'status', '?')}, data={getattr(e, 'data', '?')}. "
                              f"Stop early for {full_name}.")
                        break
                    except Exception as e:
                        # 非致命：只跳过这一条
                        print(f"  !! Failed to inspect issue {getattr(issue, 'number', '?')} "
                              f"in {full_name}: {e}")
                        continue

                    # 防御式调用 looks_mcp_related，避免单条 issue 让整个脚本崩掉
                    try:
                        if not looks_mcp_related(issue):
                            continue
                    except RetryError as e:
                        repo_had_error = True
                        print(f"  [RetryError in looks_mcp_related] {e}. "
                              f"Stop early for {full_name}.")
                        break
                    except GithubException as e:
                        repo_had_error = True
                        print(f"  [GithubException in looks_mcp_related] "
                              f"status={getattr(e, 'status', '?')}, data={getattr(e, 'data', '?')}. "
                              f"Stop early for {full_name}.")
                        break
                    except Exception as e:
                        print(f"  !! looks_mcp_related crashed for issue "
                              f"{getattr(issue, 'html_url', '?')}: {e}")
                        continue

                    row = {
                        "repo": full_name,
                        "issue_url": issue.html_url,
                        "title": issue.title or "",
                        "language": repo_lang,
                        "description": summarize_issue(issue),
                    }

                    # 根据 issue.state 写到不同文件
                    if issue.state == "open":
                        open_writer.writerow(row)
                        total_open += 1
                    elif issue.state == "closed":
                        closed_writer.writerow(row)
                        total_closed += 1
                    else:
                        # 一般不会有别的状态，但以防万一
                        print(f"  ?? Unknown issue state {issue.state} for {issue.html_url}")

            except RateLimitExceededException as e:
                repo_had_error = True
                print(f"  [RateLimitExceeded] {e}. Stop early for {full_name}.")
            except RetryError as e:
                repo_had_error = True
                print(f"  [RetryError] {e}. Stop early for {full_name}.")
            except GithubException as e:
                repo_had_error = True
                print(f"  [GithubException] status={getattr(e, 'status', '?')}, "
                      f"data={getattr(e, 'data', '?')}. Stop early for {full_name}.")

            # 只有在遍历 issues 没出错时，才把这个 repo 记为 processed
            if not repo_had_error:
                save_processed_repo(full_name)
            else:
                print(f"  !! Not marking {full_name} as processed due to errors. "
                      f"You can rerun later to 补齐 this repo.")
            time.sleep(random.uniform(1.0, 2.9))  # repo 级别降速

        print(f"\nDone.")
        print(f"  Kept {total_open} OPEN MCP issues → {OUTPUT_OPEN_ISSUES_CSV}")
        print(f"  Kept {total_closed} CLOSED MCP issues → {OUTPUT_CLOSED_ISSUES_CSV}")


if __name__ == "__main__":
    main()
