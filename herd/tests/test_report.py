"""Store -> per-cow days -> 3-day CSV -> alerts -> vet verdicts, on 10 synthetic
days of three cows and one track nobody was identified on. On the last day
cow B lies far more and cow C ruminates far less; cow A stays normal."""
import csv
import datetime as dt
import os
import random
import tempfile

import _paths  # noqa: F401
import report
from common import load_config
from store import Store


def make(folder):
    cfg = load_config(None)
    cfg["store"]["path"] = os.path.join(folder, "herd.sqlite")
    st = Store(cfg["store"]["path"])
    rng = random.Random(0)
    day0 = dt.datetime(2026, 9, 28, 6, tzinfo=dt.timezone.utc).timestamp()
    cows = {"A": (0.5, 0.35), "B": (0.45, 0.40), "C": (0.55, 0.30)}
    for d in range(10):
        for cow, (lie, rum) in cows.items():
            if d == 9 and cow == "C":
                rum = 0.10
            if d == 9 and cow == "B":
                lie = 0.80
            lie += rng.uniform(-0.03, 0.03)
            rum += rng.uniform(-0.02, 0.02)
            track, t0 = f"cam1-{d}-{cow}", day0 + d * 86400
            rows = []
            for s in range(3 * 3600):
                post = "lying" if s < lie * 3 * 3600 else "standing"
                act = "feeding" if (post == "standing" and s % 3 == 0) else "none"
                rows.append(("cam1", track, t0 + s, .1, .1, .3, .3, post, act, 1.0 if post == "lying" else 0.0))
            # the same cow seen on a second track at the same seconds (a mask edge) must count once
            if cow == "A":
                rows += [("cam2", track + "-dup", t0 + s, .1, .1, .3, .3, "standing", "none", 0.0) for s in range(60)]
                st.burst(("cam2", track + "-dup", t0, "confirmed", "A", .8, .2, .99, 0.1, "standing", "none",
                          None, 0.01, 175))
            st.seconds(rows)
            for m in range(180):
                st.burst(("cam1", track, t0 + m * 60, "confirmed" if m % 4 else "tentative", cow, .8, .2, .99,
                          0.9 if m < rum * 180 else 0.1, "lying", "none", None, 0.01, 175))
        st.seconds([("cam2", f"cam2-{d}-x", day0 + d * 86400 + s, .5, .5, .7, .7, "standing", "none", 0.0)
                    for s in range(600)])
    return cfg, st, (dt.date(2026, 9, 28) + dt.timedelta(days=9)).isoformat()


def test_csv_alerts_verdicts():
    folder = tempfile.mkdtemp()
    cfg, st, last = make(folder)
    path = report.write_period_csv(st, cfg, last, os.path.join(folder, "r.csv"))
    rows = {r["cow_id"]: r for r in csv.DictReader(open(path))}
    assert set(rows) == {"A", "B", "C", "NaN"}
    assert float(rows["A"]["observed_min"]) == 540.0           # 3 days x 3 h, the duplicate seconds once
    assert float(rows["NaN"]["observed_min"]) == 30.0          # 3 days x 600 s nobody was identified on
    a = rows["C"]
    total = sum(float(a[k]) for k in ("feeding_min", "drinking_min", "ruminating_min", "idle_min"))
    assert abs(total - float(a["observed_min"])) < 1.0, total  # every observed minute is in one activity
    alerts = report.compute_alerts(st, cfg, last)
    kinds = {(r[2], r[3]) for r in alerts}
    assert kinds == {("B", "lying_high"), ("C", "rumination_low")}, kinds
    assert len(report.compute_alerts(st, cfg, last)) == 2     # a rerun adds nothing
    report.add_verdict(st, alerts[0][0], True, "checked")
    report.add_verdict(st, alerts[1][0], False)
    stats = {s["kind"]: s for s in report.alert_stats(st)}
    assert sum(s["confirmed"] for s in stats.values()) == 1 and all(s["answered"] == 1 for s in stats.values())
