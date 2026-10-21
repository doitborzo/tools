"""Everything the system observes, in one SQLite file.

    seconds   one row per tracked cow per second: box, posture, activity
              (the once-a-second path)
    bursts    one row per cow per burst (~ once a minute): the identity decision,
              rumination (p and the yes / no the burst decided), burst
              posture / activity, lameness, quality
    events    gallery changes: cows enrolled, retired, change-over
    cannot_link   pairs of tracks seen at the same time (so two different cows)
    cow_aliases   one cow's split profile merged into another (old -> canonical)
    track_handovers   a track passed from one camera to another (mct.py)
    track_links   one camera's track joined to another's: the same cow (mct.TrackletLinker)
    alerts    raised alerts;  verdicts  the vet's answer to each

Rows are written by track, not by cow: which cow a track is gets decided from
its bursts (report.py), so a cow identified at minute 5 of a track also owns
minutes 1-4 of it.
"""

from __future__ import annotations

import os
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS seconds (cam TEXT, track TEXT, ts REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                                    posture TEXT, activity TEXT, p_lying REAL);
CREATE INDEX IF NOT EXISTS seconds_ts ON seconds(ts);
CREATE INDEX IF NOT EXISTS seconds_track ON seconds(track);
CREATE TABLE IF NOT EXISTS bursts (cam TEXT, track TEXT, ts REAL, state TEXT, cow TEXT, sim REAL, margin REAL,
                                   p REAL, rumination_p REAL, posture TEXT, activity TEXT, lameness REAL,
                                   quality REAL, n_frames INTEGER, ruminating INTEGER);
CREATE INDEX IF NOT EXISTS bursts_ts ON bursts(ts);
CREATE INDEX IF NOT EXISTS bursts_track ON bursts(track);
CREATE TABLE IF NOT EXISTS events (ts REAL, kind TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, day TEXT, cow TEXT,
                                   kind TEXT, value REAL, baseline REAL, detail TEXT,
                                   UNIQUE(day, cow, kind));
CREATE TABLE IF NOT EXISTS verdicts (alert_id INTEGER, ts REAL, confirmed INTEGER, diagnosis TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS cannot_link (track_a TEXT, track_b TEXT, ts REAL, PRIMARY KEY (track_a, track_b));
CREATE TABLE IF NOT EXISTS cow_aliases (old_cow TEXT PRIMARY KEY, canonical_cow TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS track_links (ts REAL, track TEXT PRIMARY KEY, joined_to TEXT);
CREATE TABLE IF NOT EXISTS track_handovers (ts REAL, gid TEXT, from_cam TEXT, to_cam TEXT, kind TEXT,
                                            dist_m REAL, sim REAL);
"""

BURST_COLUMNS = ("cam", "track", "ts", "state", "cow", "sim", "margin", "p", "rumination_p", "posture",
                 "activity", "lameness", "quality", "n_frames", "ruminating")


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(bursts)")}
        if "ruminating" not in cols:          # a store from before the burst wrote it
            self.db.execute("ALTER TABLE bursts ADD COLUMN ruminating INTEGER")
        self.db.commit()
        self.lock = threading.Lock()

    def seconds(self, rows):
        with self.lock:
            self.db.executemany("INSERT INTO seconds VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            self.db.commit()

    def burst(self, row):
        """cam, track, ts, state, cow, sim, margin, p, rumination_p, posture,
        activity, lameness, quality, n_frames[, ruminating (0/1)]"""
        row = tuple(row) + (None,) * (len(BURST_COLUMNS) - len(row))
        # columns named: a store made by an older version may carry extra ones (burst_quality)
        with self.lock:
            self.db.execute(f"INSERT INTO bursts ({', '.join(BURST_COLUMNS)}) "
                            f"VALUES ({', '.join('?' * len(BURST_COLUMNS))})", row)
            self.db.commit()

    def event(self, ts, kind, detail):
        with self.lock:
            self.db.execute("INSERT INTO events VALUES (?,?,?)", (ts, kind, detail))
            self.db.commit()

    def add_cannot_link(self, pairs, ts):
        """Pairs of tracks seen at the same time: two different cows."""
        rows = [(min(a, b), max(a, b), ts) for a, b in pairs if a != b]
        if rows:
            with self.lock:
                self.db.executemany("INSERT OR IGNORE INTO cannot_link VALUES (?,?,?)", rows)
                self.db.commit()

    def cannot_link_pairs(self, since=0.0):
        return {(a, b) for a, b in self.query("SELECT track_a, track_b FROM cannot_link WHERE ts >= ?", (since,))}

    def alias_cow(self, old_cow, canonical_cow, ts):
        """old_cow is canonical_cow: earlier aliases follow, the bursts are relabelled (reversible from
        the alias table: unalias_cow)."""
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO cow_aliases VALUES (?,?,?)", (old_cow, canonical_cow, ts))
            self.db.execute("UPDATE cow_aliases SET canonical_cow = ? WHERE canonical_cow = ?",
                            (canonical_cow, old_cow))
            self.db.commit()

    def unalias_cow(self, old_cow):
        with self.lock:
            self.db.execute("DELETE FROM cow_aliases WHERE old_cow = ?", (old_cow,))
            self.db.commit()

    def aliases(self):
        return dict(self.query("SELECT old_cow, canonical_cow FROM cow_aliases"))

    def link_tracks(self, ts, track, joined_to):
        """track is the same cow as joined_to (mct.TrackletLinker): reports read them as one track."""
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO track_links VALUES (?,?,?)", (ts, track, joined_to))
            self.db.commit()

    def track_roots(self):
        """track -> the track it was finally joined to (links followed to the end)."""
        links = dict(self.query("SELECT track, joined_to FROM track_links"))
        out = {}
        for t in links:
            r, seen = t, set()
            while r in links and r not in seen:
                seen.add(r)
                r = links[r]
            out[t] = r
        return out

    def handover(self, ts, gid, from_cam, to_cam, kind, dist_m=None, sim=None):
        with self.lock:
            self.db.execute("INSERT INTO track_handovers VALUES (?,?,?,?,?,?,?)",
                            (ts, gid, from_cam, to_cam, kind, dist_m, sim))
            self.db.commit()

    def query(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def execute(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur
