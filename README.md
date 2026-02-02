# MIDAS: MCP Integration Defect Analysis Set

MIDAS is a structured defect library for LLM-enabled software systems that integrate external capabilities via the Model Context Protocol (MCP).
It accompanies our empirical study of MCP integration defects and supports research on defect analysis, benchmarking, and mitigation.

---
## What are MCP-based software systems?

MCP-based software systems are software systems that employ the Model Context Protocol (MCP) to integrate external capabilities (tools, resources, and prompts) into an LLM-mediated execution loop.

It typically contains three components:
1. **MCP Host**: the AI application that manages user interaction and orchestrates the LLM-mediated execution loop.
2. **Embedded MCP Client(s)**: application-embedded mediators between the Host and each MCP Server; each client maintains a dedicated communication session to its corresponding server.
3. **MCP Server(s)**: capability providers that expose tools/resources/prompts and return structured results consumed by the Host during execution.

---
## What is inside the artifact?
Defect library and metadata for MCP-based systems studied in our empirical study.
---
### MIDAS Defect Library Overview
`mi_defects.csv` documents MCP integration defects collected from MCP-based software systems studied in our empirical study. Summary results reported in **Table 1** of our paper can be reproduced from this defect library.

Each defect entry includes:
- system/repository identifier (`repo`)
- the defect report link (`issue_url`),
- defect pattern label (`Defect Pattern`)
- defect description (`description`),
- report status as recorded in the source tracker (`status`),
- observed impact code(s) (`impact`),
- root-cause localization / component layer (`layer`, e.g., host / MCP-client / MCP-server),
- where the issue occurs (host vs server repository, if applicable) (`Host/server Repository`),
- resolution link (`resolution_url`)
- fix summary (`fix`), and relevant code location (`source_location`), when available.

---
### Studied systems / benchmark metadata

`studied_systems.csv` provides metadata for each studied MCP-based software system, including:
- full project name (`Full Name`),
- repository link (`Github URLs`),
- GitHub popularity and activity signals (e.g., `Stars`, `Number of Open Issues`, `Number of Closed Issues`),
- primary implementation language (`language`),
- brief description (`Description`),
- study-specific classification label(s) (`Classification`).

---
## How to use

Typical use cases:
- **Reproduce / extend** defect counts and pattern distributions reported in the paper.
- **Build benchmarks** for testing MCP hosts/clients/servers against known defect patterns.
- **Develop tools** for detection, diagnosis, or regression testing of MCP integration defects.
---
## License

- **Data** (`defects.csv`, `studied_systems.csv`, and other dataset files): licensed under **PDDL 1.0** (see `LICENSE-DATA`).
