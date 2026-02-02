# Repo/crawler_config.py
# MCP Repository Crawler Configuration

# ---------------------------------------------------------------------
# Search Configuration
# ---------------------------------------------------------------------
SEARCH_QUERIES = [
    # Direct MCP client searches
    '"model context protocol" client stars:>10',
    'mcp client -server stars:>10',
    '"mcp client" stars:>10',
    '@modelcontextprotocol/client stars:>10',

    # MCP with application types
    'mcp agent stars:>50',
    'mcp cli stars:>20',
    'mcp ide stars:>20',
    'mcp desktop app stars:>20',
    'mcp extension stars:>20',
    'mcp plugin stars:>20',
    'mcp chatbot stars:>20',
    'mcp assistant stars:>20',
    'mcp automation stars:>20',
    'mcp workflow stars:>20',

    # Known frameworks with MCP
    'autogen mcp stars:>100',
    'langchain mcp stars:>50',
    'semantic-kernel mcp stars:>50',
    'openai mcp stars:>50',
    'anthropic mcp stars:>50',

    # MCP configuration files
    'filename:mcp.config.json stars:>10',
    'filename:.mcp.json stars:>10',
    'path:.mcp/config.json stars:>10',

    # Code patterns
    'MCPClient language:Python stars:>20',
    'MCPClient language:TypeScript stars:>20',
    'MCPClient language:JavaScript stars:>20',
    'MCPClient language:Rust stars:>20',
    'MCPClient language:Go stars:>20',
    'StdioClientTransport stars:>10',
    'mcp.connect stars:>10',
    'ClientSession mcp stars:>10',

    # Topics
    'topic:mcp topic:client stars:>10',
    'topic:mcp topic:agent stars:>10',
    'topic:model-context-protocol stars:>10',
    'topic:mcp-client stars:>10',

    # Recent implementations
    'mcp created:>2024-01-01 stars:>50',
    '"model context protocol" created:>2024-01-01 stars:>20',
    'mcp client created:>2024-06-01 stars:>10',
]

# ---------------------------------------------------------------------
# Repository Filtering
# ---------------------------------------------------------------------
MIN_STARS = 10
MAX_REPOS_PER_QUERY = 50
CONFIDENCE_THRESHOLD = 25  # Minimum confidence score to "keep" in post-filtering

# ---------------------------------------------------------------------
# API / Crawl Configuration
# ---------------------------------------------------------------------
RATE_LIMIT_DELAY = 1.0  # Seconds between search requests
MAX_WORKERS = 5         # Parallel analysis threads

# ---------------------------------------------------------------------
# Output Configuration
# ---------------------------------------------------------------------
OUTPUT_FORMATS = ["csv", "json"]
OUTPUT_PREFIX = "mcp_clients"

# ---------------------------------------------------------------------
# Classification Thresholds
# ---------------------------------------------------------------------
CONFIDENCE_LEVELS = {
    "high_confidence": 50,
    "likely": 25,
    "possible": 10,
    "unlikely": 0,
}

# ---------------------------------------------------------------------
# Exclusion Patterns (repos to skip by name pattern)
# ---------------------------------------------------------------------
EXCLUDE_PATTERNS = [
    "mcp-server",
    "server-mcp",
    "mcp-servers",
    "awesome-mcp",    # lists / curated collections
    "mcp-examples",   # example-only repos
    "mcp-template",   # templates / boilerplate
]

# ---------------------------------------------------------------------
# Known MCP Client Repositories
# Used as seed list & ground truth for evaluation/test suite
# ---------------------------------------------------------------------
KNOWN_CLIENTS = [
    # Major frameworks with MCP integration
    "microsoft/autogen",           # Autogen agent framework with MCP
    "All-Hands-AI/OpenHands",      # Agent framework with MCP plugin support
    "QwenLM/Qwen-Agent",           # Qwen Agent framework with MCP

    # IDE extensions / dev tools
    "continuedev/continue",        # VS Code extension with dynamic MCP loading
    "anthropics/claude-code",      # Claude code assistant with MCP

    # Automation / workflow / browser hosts
    "browser-use/browser-use",     # Browser automation with MCP
    "activepieces/activepieces",   # Workflow automation platform with MCP

    # Chat & AI client hosts
    "daodao97/chatmcp",            # Dedicated MCP chat client
    "NitroRCr/AIaW",               # AI workspace with MCP support,
    # optionally:
    "lastmile-ai/mcp-agent",
    "LSTM-Kirigaya/openmcp-client",
    "CoderGamester/mcp-unity",
]

# ---------------------------------------------------------------------
# Evidence Weights
# Tuned to reflect relative importance of each signal category.
# ---------------------------------------------------------------------
EVIDENCE_WEIGHTS = {
    "package_dependency": 30,
    "config_file": 25,
    "client_code_pattern": 20,
    "description_mention": 15,
    "readme_mention": 15,
    "topic_mention": 10,
    "framework_integration": 10,   # reserved if you add framework-specific boosts
    "ai_classification": 20,       # delta from AI confirmation/rejection
}

# ---------------------------------------------------------------------
# AI Assist Toggle
# If True and OPENAI_API_KEY is set, the crawler will use LLM to refine classification.
# For strict deterministic runs, set this to False.
# ---------------------------------------------------------------------
USE_AI_ASSIST = True
