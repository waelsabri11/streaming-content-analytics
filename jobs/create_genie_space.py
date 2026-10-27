"""Create a Genie space (natural-language questions) over the gold tables via the Databricks CLI."""

import json
import subprocess
import uuid

TABLES = {
    "workspace.gold.netflix_cohort": "Netflix titles released since 2023 with pre-release features; y = log(views in release semester)",
    "workspace.gold.netflix_life_curves": "Title-semester points: views per exposure day (rate) vs days since release (t_mid); at_floor marks views at the 100k reporting floor",
    "workspace.gold.netflix_performance_score": "Out-of-sample predicted log views (pred) with 80% interval (lo, hi) and the actual y, per test semester fold",
    "workspace.gold.netflix_value_index": "Cumulative views over 4 semesters since release (value_views) and index vs median of the same kind",
    "workspace.gold.twitch_channel_daily": "Daily viewers per Twitch channel: viewers, new_viewers, returning_viewers (day = day of the 43-day window)",
    "workspace.gold.twitch_viewer_segments": "Per-viewer segment assignment and behavioural features of 15.5M Twitch viewers",
    "workspace.silver.twitch_streamers": "Channel name dictionary: streamer_id <-> streamer",
}
QUESTIONS = [
    "How many viewers and how much watch time does each viewer segment have?",
    "Which Netflix titles have the highest value index?",
    "What is the median views per exposure day by kind and semesters since release?",
    "How well did the 80% interval cover the actual views in each test semester?",
    "Which Twitch channels had the most viewer-days?",
]


def hid():
    return uuid.uuid4().hex


space = {
    "version": 2,
    "config": {"sample_questions": sorted([{"id": hid(), "question": [q]} for q in QUESTIONS], key=lambda x: x["id"])},
    "data_sources": {"tables": [{"identifier": k, "description": [v]} for k, v in sorted(TABLES.items())]},
    "instructions": {"text_instructions": [{"id": hid(), "content": [
        "Twitch data is a proxy for streaming consumption (chat presence); slots are 10 minutes; hours = slots * 10 / 60. "
        "Netflix views are rounded to 100k; semester = half-year; rel_sem is the release semester. Never invent numbers: answer only from the tables."]}]},
}
out = subprocess.check_output(["databricks", "genie", "create-space", "49d284e9ba6c044b", json.dumps(space), "--title", "Streaming content analytics", "--description",
                               "Ask questions about Twitch viewer segments, channel audiences and Netflix scores / value index", "-o", "json"])
print(out.decode()[:500])
