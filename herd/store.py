"""Everything the system observes, in one SQLite file.

    seconds   one row per tracked cow per second: box, posture, activity
              (the once-a-second path)
    bursts    one row per cow per burst (~ once a minute): the identity decision,
              rumination, burst posture / activity, lameness, quality
    events    gallery changes: cows enrolled, retired, change-over
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
                                   quality REAL, n_frames INTEGER);
CREATE INDEX IF NOT EXISTS bursts_ts ON bursts(ts);
CREATE INDEX IF NOT EXISTS bursts_track ON bursts(track);
CREATE TABLE IF NOT EXISTS events (ts REAL, kind TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, day TEXT, cow TEXT,
                                   kind TEXT, value REAL, baseline REAL, detail TEXT,
                                   UNIQUE(day, cow, kind));
CREATE TABLE IF NOT EXISTS verdicts (alert_id INTEGER, ts REAL, confirmed INTEGER, diagnosis TEXT, note TEXT);
"""


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def seconds(self, rows):
        with self.lock:
            self.db.executemany("INSERT INTO seconds VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            self.db.commit()

    def burst(self, row):
        with self.lock:
            self.db.execute("INSERT INTO bursts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            self.db.commit()

    def event(self, ts, kind, detail):
        with self.lock:
            self.db.execute("INSERT INTO events VALUES (?,?,?)", (ts, kind, detail))
            self.db.commit()

    def query(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def execute(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur
