"""Record the machine/software that produced the results (reports/results/environment.json)."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import delta
import pyspark

root = Path(__file__).resolve().parents[1]
mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode()) if sys.platform == "darwin" else 0
env = {
    "platform": f"{platform.system()} {platform.machine()}",
    "cpus": os.cpu_count(),
    "ram_gb": round(mem / 2**30),
    "spark": pyspark.__version__,
    "delta": getattr(delta, "__version__", "4.0.0"),
    "python": platform.python_version(),
}
(root / "reports" / "results" / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
print(env)
