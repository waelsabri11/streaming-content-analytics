import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    os.environ["SCA_WAREHOUSE"] = str(tmp_path_factory.mktemp("wh"))
    from pyspark.sql import SparkSession

    s = (
        SparkSession.builder.master("local[2]")
        .appName("tests")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield s
    s.stop()
