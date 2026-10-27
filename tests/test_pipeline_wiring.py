"""The Lakeflow file cannot run without Databricks; check it registers what we expect."""

import builtins
import sys
import types
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class _Conf:
    def get(self, key, default=None):
        return {"twitch.raw_path": "/Volumes/x/bronze/raw/twitch/"}.get(key, default or str(ROOT / "src"))


def _load_pipeline():
    registered = {}

    def deco(kind):
        def outer(*a, **kw):
            name_kw = kw.get("name")

            def wrap(fn):
                registered.setdefault(name_kw or fn.__name__, {"kind": kind, "expect": []})
                registered[name_kw or fn.__name__]["fn"] = fn
                return fn

            return wrap
        return outer

    def expect(kind):
        def outer(name, expr):
            def wrap(fn):
                reg = registered.setdefault(fn.__name__, {"kind": "?", "expect": []})
                reg["expect"].append((kind, name, expr))
                return fn
            return wrap
        return outer

    dp = types.ModuleType("pyspark.pipelines")
    dp.table, dp.materialized_view = deco("table"), deco("mv")
    dp.expect, dp.expect_or_drop, dp.expect_or_fail = expect("warn"), expect("drop"), expect("fail")
    dp.read = lambda n: n
    import pyspark

    sys.modules["pyspark.pipelines"] = dp
    pyspark.pipelines = dp
    builtins.spark = types.SimpleNamespace(conf=_Conf())
    src = (ROOT / "pipelines" / "twitch_pipeline.py").read_text()
    exec(compile(src, "twitch_pipeline.py", "exec"), {"__name__": "pipeline"})
    return registered


def test_pipeline_registers_datasets_and_expectations():
    reg = _load_pipeline()
    # decorators order: expectations are applied bottom-up, then the dataset decorator by name
    names = {fn.__name__ for fn in (v["fn"] for v in reg.values() if "fn" in v)}
    assert {"twitch_interactions", "twitch_views", "twitch_view_sessions"} <= names
    kinds = {k for v in reg.values() for k, _, _ in v["expect"]}
    assert {"drop", "fail", "warn"} <= kinds


def test_bundle_references_exist():
    b = yaml.safe_load((ROOT / "databricks.yml").read_text())
    assert b["bundle"]["name"] == "streaming-content-analytics"
    pipe = yaml.safe_load((ROOT / "resources" / "twitch_pipeline.yml").read_text())["resources"]["pipelines"]["twitch_silver"]
    for lib in pipe["libraries"]:
        assert (ROOT / "resources" / lib["file"]["path"]).resolve().exists()
    all_jobs = yaml.safe_load((ROOT / "resources" / "jobs.yml").read_text())["resources"]["jobs"]
    for jobs in all_jobs.values():
        for task in jobs["tasks"]:
            if "spark_python_task" in task:
                assert (ROOT / "resources" / task["spark_python_task"]["python_file"]).resolve().exists()
                assert task["environment_key"] == "default"
        keys = {t["task_key"] for t in jobs["tasks"]}
        for t in jobs["tasks"]:
            for dep in t.get("depends_on", []):
                assert dep["task_key"] in keys
