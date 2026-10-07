"""From the store to what the farm sees: who is who, per-cow times, the
3-day CSV, alerts for the vet, and how right those alerts were.

Which cow a track is: the cow most of its confirmed bursts say (at least
two thirds of them), else the track stays unidentified - its time goes to
the NaN row, never to a guessed cow. Seconds of one cow seen on two tracks
at once (a mask edge) count once.

Times per cow and day (farm local time):
  lying, standing                   seconds of the 1 fps path
  feeding, drinking                 seconds of the 1 fps path
  ruminating                        the share of her bursts that say ruminating,
                                    times her observed minutes (bursts sample
                                    rumination once a minute)
  idle ("did nothing")              the rest: neither feeding, drinking nor ruminating
  lameness                          mean score of her bursts and how many there were
                                    (empty until a lameness head is trained)
  observed                          minutes she was seen at all - cameras do not see
                                    the whole barn, so compare shares, not hours

Alerts compare a cow with herself: her median over the previous days and how
much she normally varies (MAD). Rumination: a drop. Lying: a drop, a rise or
both (alerts.lying_directions) - which way is "worse" is the vet's call.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import sys

from common import load_config
from store import Store

FIELDS = ("observed_min", "lying_min", "standing_min", "feeding_min", "drinking_min", "ruminating_min",
          "idle_min", "lying_share", "rumination_share", "lameness_score", "lameness_bursts", "bursts")


def day_expr(cfg):
    off = cfg["farm"]["timezone_offset_hours"] * 3600
    return f"date(ts + {off}, 'unixepoch')"


def track_map(store, min_share=2 / 3):
    """track -> cow from its confirmed bursts; tracks without a clear majority are left out."""
    votes = {}
    for track, cow, n in store.query("SELECT track, cow, COUNT(*) FROM bursts WHERE state='confirmed' "
                                     "GROUP BY track, cow"):
        votes.setdefault(track, {})[cow] = n
    out = {}
    for track, v in votes.items():
        cow, n = max(v.items(), key=lambda kv: kv[1])
        if n >= min_share * sum(v.values()):
            out[track] = cow
    return out


def daily(store, cfg, since=None, until=None):
    """{(cow, day): {field: value}}; cow None = the unidentified (NaN) row."""
    tm = track_map(store)
    store.execute("DROP TABLE IF EXISTS temp.trackmap")
    store.execute("CREATE TEMP TABLE trackmap (track TEXT PRIMARY KEY, cow TEXT)")
    with store.lock:
        store.db.executemany("INSERT INTO temp.trackmap VALUES (?,?)", tm.items())
        store.db.commit()
    day = day_expr(cfg)
    where = []
    if since is not None:
        where.append(f"s.ts >= {float(since)}")
    if until is not None:
        where.append(f"s.ts < {float(until)}")
    w = ("WHERE " + " AND ".join(where)) if where else ""
    sec = lambda cond: f"COUNT(DISTINCT CASE WHEN {cond} THEN CAST(s.ts AS INTEGER) END)"
    rows = store.query(f"""
        SELECT m.cow, {day.replace('ts', 's.ts')} AS d, COUNT(DISTINCT CAST(s.ts AS INTEGER)),
               {sec("s.posture='lying'")}, {sec("s.posture='standing'")},
               {sec("s.activity='feeding'")}, {sec("s.activity='drinking'")}, {sec("s.activity='none'")}
        FROM seconds s LEFT JOIN temp.trackmap m ON s.track = m.track {w}
        GROUP BY m.cow, d""")
    # unidentified tracks overlap in time, so their seconds are counted per track
    nan_rows = store.query(f"""
        SELECT {day.replace('ts', 's.ts')} AS d, COUNT(*) FROM seconds s LEFT JOIN temp.trackmap m
        ON s.track = m.track {w + (' AND' if w else 'WHERE')} m.cow IS NULL GROUP BY d""")
    bw = w.replace("s.ts", "b.ts")
    bursts = store.query(f"""
        SELECT m.cow, {day.replace('ts', 'b.ts')} AS d, COUNT(*), SUM(b.rumination_p >= 0.5),
               AVG(b.lameness), SUM(b.lameness IS NOT NULL)
        FROM bursts b JOIN temp.trackmap m ON b.track = m.track {bw} GROUP BY m.cow, d""")
    out = {}
    for cow, d, obs, lying, standing, feed, drink, none in rows:
        if cow is None:
            continue
        out[(cow, d)] = {"observed_min": obs / 60, "lying_min": lying / 60, "standing_min": standing / 60,
                         "feeding_min": feed / 60, "drinking_min": drink / 60, "_none_min": none / 60}
    for d, n in nan_rows:
        out[(None, d)] = {"observed_min": n / 60}
    for cow, d, n, rum, lame, nlame in bursts:
        r = out.setdefault((cow, d), {"observed_min": 0.0})
        r["bursts"] = n
        r["rumination_share"] = (rum or 0) / n if n else None
        r["lameness_score"] = lame
        r["lameness_bursts"] = nlame or 0
    for (cow, d), r in out.items():
        if cow is None:
            continue
        obs = r.get("observed_min") or 0
        share = r.get("rumination_share")
        r["ruminating_min"] = share * obs if share is not None else None
        r["idle_min"] = max(0.0, r.pop("_none_min", 0) - (r["ruminating_min"] or 0))
        r["lying_share"] = r.get("lying_min", 0) / obs if obs else None
    return out


def names(cfg):
    path = os.path.join(os.path.dirname(os.path.abspath(cfg["store"]["path"])), "names.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def write_period_csv(store, cfg, end_day, out_path, days=None):
    """Sum of the last `days` days up to end_day (inclusive), one row per cow + NaN."""
    days = days or cfg["report"]["period_days"]
    end = dt.date.fromisoformat(end_day)
    wanted = {(end - dt.timedelta(days=i)).isoformat() for i in range(days)}
    per = daily(store, cfg)
    label = names(cfg)
    tot = {}
    for (cow, d), r in per.items():
        if d not in wanted:
            continue
        t = tot.setdefault(cow, {"_rum_bursts": 0.0})
        for k in ("observed_min", "lying_min", "standing_min", "feeding_min", "drinking_min", "ruminating_min",
                  "idle_min", "bursts", "lameness_bursts"):
            if r.get(k) is not None:
                t[k] = t.get(k, 0) + r[k]
        if r.get("lameness_score") is not None and r.get("lameness_bursts"):
            t["_lame_sum"] = t.get("_lame_sum", 0) + r["lameness_score"] * r["lameness_bursts"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["period_start", "period_end", "cow_id", "label", *FIELDS])
        start = min(wanted)
        for cow in sorted(tot, key=lambda c: (c is None, c or "")):
            t = tot[cow]
            obs = t.get("observed_min", 0)
            row = {**t, "lying_share": t.get("lying_min", 0) / obs if obs and cow else None,
                   "rumination_share": t.get("ruminating_min", 0) / obs if obs and cow and t.get("ruminating_min") is not None else None,
                   "lameness_score": t["_lame_sum"] / t["lameness_bursts"] if t.get("lameness_bursts") else None}
            w.writerow([start, end_day, cow if cow else "NaN", label.get(cow, "") if cow else "unidentified",
                        *[("" if row.get(f) is None else round(row[f], 3)) for f in FIELDS]])
    return out_path


def compute_alerts(store, cfg, day):
    """Alerts for one day against each cow's own baseline; stored (once) and returned."""
    a = cfg["alerts"]
    per = daily(store, cfg)
    target = dt.date.fromisoformat(day)
    base_days = {(target - dt.timedelta(days=i)).isoformat() for i in range(1, a["baseline_days"] + 1)}
    out = []
    for (cow, d), r in per.items():
        if cow is None or d != day or (r.get("observed_min") or 0) < a["min_observed_minutes"]:
            continue
        hist = [per[(cow, b)] for b in base_days if (cow, b) in per
                and (per[(cow, b)].get("observed_min") or 0) >= a["min_observed_minutes"]]
        if len(hist) < a["min_baseline_days"]:
            continue
        checks = [("rumination_low", "rumination_share", "low", a["rumination_drop"])]
        checks += [(f"lying_{d_}", "lying_share", d_, a["lying_change"]) for d_ in a["lying_directions"]]
        for kind, field, direction, change in checks:
            today = r.get(field)
            past = [h[field] for h in hist if h.get(field) is not None]
            if today is None or len(past) < a["min_baseline_days"]:
                continue
            med = statistics.median(past)
            mad = statistics.median([abs(x - med) for x in past]) * 1.4826
            z = (today - med) / max(mad, 0.02)
            hit = (today <= med * (1 - change) and z <= -a["mad_z"]) if direction == "low" else \
                  (today >= med * (1 + change) and z >= a["mad_z"])
            if hit:
                detail = f"{field} {today:.2f} vs her median {med:.2f} over {len(past)} days (z {z:.1f})"
                out.append((day, cow, kind, today, med, detail))
        if cfg["lameness"]["enabled"] and r.get("lameness_score") is not None and \
                r["lameness_score"] >= cfg["lameness"].get("alert_score", 3.0):
            out.append((day, cow, "lame", r["lameness_score"], None,
                        f"lameness score {r['lameness_score']:.1f} over {r.get('lameness_bursts')} walking bursts"))
    now = dt.datetime.now().timestamp()
    for day_, cow, kind, val, base, detail in out:
        store.execute("INSERT OR IGNORE INTO alerts (created, day, cow, kind, value, baseline, detail) "
                      "VALUES (?,?,?,?,?,?,?)", (now, day_, cow, kind, val, base, detail))
    return store.query("SELECT id, day, cow, kind, round(value, 3), round(baseline, 3), detail FROM alerts "
                       "WHERE day = ? ORDER BY cow, kind", (day,))


def add_verdict(store, alert_id, confirmed, diagnosis="", note=""):
    store.execute("INSERT INTO verdicts VALUES (?,?,?,?,?)",
                  (alert_id, dt.datetime.now().timestamp(), 1 if confirmed else 0, diagnosis, note))


def alert_stats(store):
    """Per kind: alerts, answered by the vet, confirmed, and the share that were right."""
    rows = store.query("""SELECT a.kind, COUNT(DISTINCT a.id), COUNT(v.alert_id), SUM(v.confirmed)
                          FROM alerts a LEFT JOIN verdicts v ON v.alert_id = a.id GROUP BY a.kind""")
    return [{"kind": k, "alerts": n, "answered": ans, "confirmed": conf or 0,
             "precision": (conf or 0) / ans if ans else None} for k, n, ans, conf in rows]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("csv", help="the 3-day (report.period_days) CSV")
    r.add_argument("--end", default=dt.date.today().isoformat())
    r.add_argument("--out", default=None)
    a = sub.add_parser("alerts", help="compute and list alerts for a day")
    a.add_argument("--day", default=dt.date.today().isoformat())
    v = sub.add_parser("verdict", help="the vet's answer to an alert")
    v.add_argument("--alert", type=int, required=True)
    v.add_argument("--confirmed", choices=("yes", "no"), required=True)
    v.add_argument("--diagnosis", default="")
    v.add_argument("--note", default="")
    sub.add_parser("alert-stats", help="how often alerts were right, per kind")
    d = sub.add_parser("daily", help="per cow per day, as JSON lines")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    store = Store(cfg["store"]["path"])
    if args.cmd == "csv":
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(cfg["store"]["path"])),
                                       f"report_{args.end}.csv")
        print(write_period_csv(store, cfg, args.end, out))
    elif args.cmd == "alerts":
        for row in compute_alerts(store, cfg, args.day):
            print("#{}  {}  cow {}  {}  {} (baseline {})  {}".format(*row))
    elif args.cmd == "verdict":
        add_verdict(store, args.alert, args.confirmed == "yes", args.diagnosis, args.note)
        print("saved")
    elif args.cmd == "alert-stats":
        for s in alert_stats(store):
            print(json.dumps(s))
    elif args.cmd == "daily":
        for (cow, day_), row in sorted(daily(store, cfg).items(), key=lambda kv: (kv[0][1], kv[0][0] or "")):
            print(json.dumps({"cow": cow, "day": day_, **{k: (round(v, 3) if isinstance(v, float) else v)
                                                          for k, v in row.items()}}))


if __name__ == "__main__":
    sys.exit(main())
