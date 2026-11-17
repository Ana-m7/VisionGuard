"""Per-dataset download + evaluate modules. Each module exposes a single
entry point, ``run(args) -> dict``, returning a JSON-serialisable result
with at least a ``status`` field: "OK", "SKIPPED: <reason>", or
"PARTIAL: <reason>" — never a fabricated metric for a dataset that failed
to download.
"""
