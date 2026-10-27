"""Ask the Genie space the sample questions through the REST API and record answer latency + generated SQL."""

import json
import subprocess
import sys
import time
from pathlib import Path

SPACE = sys.argv[1] if len(sys.argv) > 1 else "01f1d20c1c861beabd466c007ee611fd"
QUESTIONS = [
    "How many viewers and how much watch time does each viewer segment have?",
    "Which Netflix titles have the highest value index?",
    "What is the median views per exposure day by kind and semesters since release?",
    "How well did the 80% interval cover the actual views in each test semester?",
    "Which Twitch channels had the most viewer-days?",
]


def api(method, path, body=None):
    cmd = ["databricks", "api", method, path] + (["--json", json.dumps(body)] if body else [])
    return json.loads(subprocess.check_output(cmd))


out = []
for q in QUESTIONS:
    t0 = time.time()
    r = api("post", f"/api/2.0/genie/spaces/{SPACE}/start-conversation", {"content": q})
    cid, mid = r["conversation_id"], r["message_id"]
    while True:
        m = api("get", f"/api/2.0/genie/spaces/{SPACE}/conversations/{cid}/messages/{mid}")
        if m["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(2)
        if time.time() - t0 > 300:
            break
    sql = next((a["query"]["query"] for a in m.get("attachments", []) if "query" in a), None)
    out.append({"question": q, "status": m["status"], "seconds": round(time.time() - t0, 1), "sql": sql})
    print(out[-1]["status"], out[-1]["seconds"], q)
Path("reports/results/databricks/genie.json").write_text(json.dumps({"space_id": SPACE, "answers": out}, indent=1))
