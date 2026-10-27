"""Dashboard query latency: Databricks SQL warehouse vs local Spark SQL on the same gold tables."""

import importlib.machinery
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "jobs"))
import create_dashboard as cd  # noqa: E402

loader = importlib.machinery.SourceFileLoader("dbsql", str(Path.home() / ".local/bin/dbsql"))
spec = importlib.util.spec_from_loader("dbsql", loader)
dbsql = importlib.util.module_from_spec(spec)
loader.exec_module(dbsql)

REPS = 5
res = {"queries": {}, "reps": REPS}
# warehouse: stop-state cold start measured on the first query only if the warehouse was stopped
t0 = time.perf_counter()
dbsql.run("SELECT 1")
res["databricks_cold_first_query_seconds"] = round(time.perf_counter() - t0, 2)
from streaming_content.spark import get_spark  # noqa: E402

spark = get_spark("sqllat", driver_mem="8g", cores="4")
for k, (_, q) in cd.DATASETS.items():
    dq = q.format(c="workspace")
    lq = q.replace("{c}.gold.", "gold.").replace("{c}.silver.", "silver.")
    dbsql.run(dq)  # warm-up
    spark.sql(lq).collect()
    d, loc = [], []
    for _ in range(REPS):
        d.append(dbsql.run(dq)[1])
        t1 = time.perf_counter()
        spark.sql(lq).collect()
        loc.append(time.perf_counter() - t1)
    res["queries"][k] = {"databricks_p50": float(np.percentile(d, 50)), "databricks_p95": float(np.percentile(d, 95)), "databricks_runs": [round(x, 2) for x in d],
                         "local_p50": float(np.percentile(loc, 50)), "local_p95": float(np.percentile(loc, 95)), "local_runs": [round(x, 2) for x in loc]}
    print(k, res["queries"][k]["databricks_p50"], res["queries"][k]["local_p50"])
res["note"] = "Databricks timings include the REST round trip from a Mac in Europe; local Spark timings are in-process on warm tables. Local tables come from the local run (same code, slightly different segmentation k)."
(ROOT / "reports/results/sql_latency.json").write_text(json.dumps(res, indent=1))
