"""Per-request latency of the Model Serving endpoint (sequential requests, one row each), via the Databricks SDK."""

import importlib.machinery
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from databricks.sdk import WorkspaceClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from streaming_content.netflix.features import FEATURES_CAT, FEATURES_NUM  # noqa: E402

loader = importlib.machinery.SourceFileLoader("dbsql", str(Path.home() / ".local/bin/dbsql"))
spec = importlib.util.spec_from_loader("dbsql", loader)
dbsql = importlib.util.module_from_spec(spec)
loader.exec_module(dbsql)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
cols = FEATURES_NUM + FEATURES_CAT
rows, _ = dbsql.run(f"SELECT {', '.join('`' + c + '`' for c in cols)} FROM workspace.gold.netflix_cohort WHERE rel_sem = '2026H1' LIMIT 200")
recs = [{k: (None if v is None else (float(v) if k in FEATURES_NUM else v)) for k, v in r.items()} for r in rows]
w = WorkspaceClient()
lat, errors = [], 0
first_err = None
for i in range(N):
    t0 = time.perf_counter()
    try:
        w.serving_endpoints.query("sca-netflix-score", dataframe_records=[recs[i % len(recs)]])
        lat.append(time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001
        errors += 1
        first_err = first_err or f"{type(e).__name__}: {str(e)[:300]}"
a = np.array(lat)
out = {"endpoint": "sca-netflix-score", "requests": N, "errors": errors, "first_error": first_err, "client": "databricks-sdk from a Mac in Europe, sequential, 1 row per request",
       "first_request_seconds": float(a[0]) if len(a) else None,
       **({"p50_s": float(np.percentile(a[1:], 50)), "p95_s": float(np.percentile(a[1:], 95)), "p99_s": float(np.percentile(a[1:], 99)), "min_s": float(a[1:].min())} if len(a) > 2 else {})}
Path("reports/results/databricks/serving_latency.json").write_text(json.dumps(out, indent=1))
print(out)
