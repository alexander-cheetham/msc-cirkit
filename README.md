## Development

Install dependencies with uv:

```bash
uv venv
uv pip install -e .[test]
```

Install optional graph visualization dependencies:

```bash
uv pip install -e .[graphviz]
```

Run tests:

```bash
.venv/bin/pytest -q
```

