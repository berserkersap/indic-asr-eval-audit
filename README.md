# indic-asr-eval-audit

Auditing how Indic ASR is evaluated: number-aware metrics and synthetic-vs-real test sets.

> Work in progress. Results and the research log are added slice by slice.

## Setup

Requires Python 3.12+ (developed on 3.14, Windows 11, RTX 5060 8 GB).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1

# Core metrics and analysis (CPU only)
pip install -e ".[dev]"

# ASR and synthesis (GPU). Install a CUDA build of torch first; cu128 or newer
# is needed for RTX 50-series (Blackwell) GPUs.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[asr,synth,dev]"
```

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest
pre-commit install                   # run the same checks on every commit
```

## License

Code is MIT. Datasets keep their own licences; see the data section once added.
