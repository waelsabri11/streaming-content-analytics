"""Export the small tables the Streamlit app needs to reports/app_data (parquet/json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from streaming_content import config  # noqa: E402
from streaming_content.spark import get_spark  # noqa: E402

OUT = config.REPORTS / "app_data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spark = get_spark("export", driver_mem="8g", cores="4")
    def t(layer, n):
        df = spark.table(config.table(layer, n))
        # timestamps -> ISO strings: Spark's tz-aware toPandas path fails on this pandas/tzdata combination
        return df.select(*[F.col(c).cast("string") if "timestamp" in d.lower() else F.col(c) for c, d in df.dtypes])

    t("gold", "netflix_performance_score").toPandas().assign(release_date=lambda d: pd.to_datetime(d["release_date"])).to_parquet(OUT / "netflix_scores.parquet")
    t("gold", "netflix_life_curves").toPandas().to_parquet(OUT / "netflix_curves.parquet")
    t("gold", "netflix_value_index").toPandas().to_parquet(OUT / "netflix_value_index.parquet")
    nf = json.loads((config.RESULTS_DIR / "netflix.json").read_text())
    shapes = {k: {"power_g": v["power_gamma"], "tau": v["power_tau_days"]} for k, v in nf["life_curves"]["by_kind"].items()}
    (OUT / "netflix_shapes.json").write_text(json.dumps(shapes))

    seg = json.loads((config.RESULTS_DIR / "twitch_segments.json").read_text())
    prof = pd.DataFrame(seg["profile"])
    prof.to_parquet(OUT / "twitch_segment_profile.parquet")
    t("gold", "twitch_segment_channel_affinity").toPandas().to_parquet(OUT / "twitch_segment_affinity.parquet")

    # top 200 channels by viewer-days: daily new vs returning viewers
    cd = t("gold", "twitch_channel_daily")
    top = cd.groupBy("streamer_id").agg(F.sum("viewers").alias("vd")).orderBy(F.desc("vd")).limit(200)
    dim = t("silver", "twitch_streamers")
    (cd.join(F.broadcast(top.select("streamer_id")), "streamer_id").join(dim, "streamer_id")
       .select("streamer", "day", "viewers", "new_viewers", "returning_viewers").toPandas()
       .to_parquet(OUT / "twitch_channel_daily_top.parquet"))
    print("exported to", OUT)


if __name__ == "__main__":
    main()
