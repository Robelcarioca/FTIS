# FTIS Research Assets

This folder contains publication and portfolio assets:

- `ftis_research_paper.md`: IEEE-style research draft.
- `technical_whitepaper.md`: implementation and architecture whitepaper.
- `benchmark_report.md`: model benchmark and monitoring report scaffold.
- `figures/*.mmd`: Mermaid architecture and workflow figures.

To create PDF outputs, render the Markdown files with a local Markdown/PDF tool such as Pandoc:

```bash
pandoc docs/research/ftis_research_paper.md -o docs/research/ftis_research_paper.pdf
pandoc docs/research/technical_whitepaper.md -o docs/research/technical_whitepaper.pdf
pandoc docs/research/benchmark_report.md -o docs/research/benchmark_report.pdf
```
