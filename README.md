# rarg-ray-patterns

Common Ray distributed computing patterns for RARG codebases

## Development
This package is managed with [uv](https://docs.astral.sh/uv/).

- Install: `uv sync`
- Test: `uv run --group test pytest tests/`
- Pre-commit hooks gate commits: `uv run --group dev pre-commit run -a`
- Versioning is via `tbump` — never hand-edit version strings.
