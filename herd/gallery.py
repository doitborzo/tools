"""Who is this cow: the herd's appearance-and-gait memory.

Every cow in the gallery is a handful of prototypes - k-means centres of her
own confirmed fingerprints over the last week (lying, walking, dirty, clean
look different; one average would blur them). A burst's fingerprint is
matched against every prototype:

    best cow, its similarity, the margin to the best OTHER cow, the burst's
    quality  ->  the reliability model (abstain.py)  ->  p(right)

    p >= cut-off                      confirmed  (cow id)
    similar to someone, p < cut-off   tentative  (counts as NaN in reports)
    similar to no one                 unknown    -> pool of candidates for new cows

No RFID and no list of new cows: the network names cows itself. The
unknown pool is grouped (bursts of one track are one cow, so tracks are
grouped, not bursts); a group seen in enough bursts becomes a new cow
"new-<date>-<n>". On the very first start the gallery is empty and the
whole herd is enrolled this way. Every Tuesday 10:00 (configurable) cows
change: from then the gallery expects new faces, and cows not seen within
identity.retire_after_hours of it are retired. Prototypes are rebuilt
nightly from confirmed fingerprints only, so a mistake does not teach the
gallery to repeat it. An optional names.json maps ids to ear-tag numbers for
the reports; nothing depends on it.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np

import abstain as abstain_mod


def spherical_kmeans(X, k, iters=30, seed=0):
    """k-means on the unit sphere (cosine), k-means++ start. X (n, d) normalised."""
    rng = np.random.default_rng(seed)
    n = len(X)
    k = max(1, min(k, n))
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d = 1 - np.max(X @ np.stack(centers).T, 1)
        d = np.clip(d, 0, None) ** 2
        centers.append(X[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else X[rng.integers(n)])
    C = np.stack(centers)
    for _ in range(iters):
        a = np.argmax(X @ C.T, 1)
        newC = np.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        newC /= np.linalg.norm(newC, axis=1, keepdims=True) + 1e-9
        if np.allclose(newC, C, atol=1e-5):
            break
        C = newC
    return C


def _norm(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class Gallery:
    def __init__(self, folder, cfg, abstain=None):
        self.folder = folder
        self.cfg = cfg["identity"]
        self.farm = cfg["farm"]
        self.abstain = abstain              # dict from abstain.json, or None (margin rule only)
        self.cows = {}                      # id -> {"prototypes", "created", "last_seen"}
        self.history = {}                   # id -> [(ts, fingerprint)]
        self.pool = []                      # unknown: {"ts", "track", "fp"}
        self.migration_since = None
        self.next_new = {}
        os.makedirs(folder, exist_ok=True)
        self._load()

    # ------------------------------------------------------------- persistence
    def _paths(self):
        return os.path.join(self.folder, "gallery.json"), os.path.join(self.folder, "gallery.npz")

    def _load(self):
        js, nz = self._paths()
        if not os.path.exists(js):
            return
        meta = json.load(open(js, encoding="utf-8"))
        arr = np.load(nz) if os.path.exists(nz) else {}
        for cid, c in meta["cows"].items():
            self.cows[cid] = {"prototypes": arr[f"p_{cid}"], "created": c["created"], "last_seen": c["last_seen"]}
            if f"h_{cid}" in arr:
                self.history[cid] = list(zip(arr[f"ht_{cid}"].tolist(), arr[f"h_{cid}"]))
        if "pool_fp" in arr:
            self.pool = [{"ts": t, "track": tr, "fp": f} for t, tr, f in
                         zip(arr["pool_ts"].tolist(), meta["pool_tracks"], arr["pool_fp"])]
        self.migration_since = meta.get("migration_since")
        self.next_new = meta.get("next_new", {})

    def save(self):
        js, nz = self._paths()
        arrays = {}
        for cid, c in self.cows.items():
            arrays[f"p_{cid}"] = c["prototypes"]
            h = self.history.get(cid, [])
            if h:
                arrays[f"ht_{cid}"] = np.array([t for t, _ in h])
                arrays[f"h_{cid}"] = np.stack([f for _, f in h])
        if self.pool:
            arrays["pool_ts"] = np.array([p["ts"] for p in self.pool])
            arrays["pool_fp"] = np.stack([p["fp"] for p in self.pool])
        np.savez(nz + ".part.npz", **arrays)
        os.replace(nz + ".part.npz", nz)
        meta = {"cows": {cid: {"created": c["created"], "last_seen": c["last_seen"],
                               "prototypes": len(c["prototypes"])} for cid, c in self.cows.items()},
                "pool_tracks": [p["track"] for p in self.pool], "migration_since": self.migration_since,
                "next_new": self.next_new}
        with open(js + ".part", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        os.replace(js + ".part", js)

    # ------------------------------------------------------------------ match
    def match(self, fp, quality_max=0.0, quality_mean=0.0, area=0.05):
        """-> {"state": confirmed|tentative|unknown, "cow", "sim", "margin", "p"}."""
        fp = _norm(fp)
        if not self.cows:
            return {"state": "unknown", "cow": None, "sim": None, "margin": None, "p": None}
        ids = list(self.cows)
        best_per_cow = np.array([float(np.max(self.cows[c]["prototypes"] @ fp)) for c in ids])
        order = np.argsort(-best_per_cow)
        best, sim = ids[order[0]], float(best_per_cow[order[0]])
        second = float(best_per_cow[order[1]]) if len(ids) > 1 else -1.0
        margin = sim - second
        if self.abstain:
            row = {"sim": sim, "margin": margin, "quality_max": quality_max, "quality_mean": quality_mean,
                   "area": area}
            feats = tuple(self.abstain.get("features", abstain_mod.FEATURES))
            p = float(abstain_mod.predict(self.abstain, abstain_mod.feature_matrix([row], feats))[0])
            sure = p >= self.abstain["threshold"]
        else:                                # no reliability model yet: a plain margin rule
            p = None
            sure = margin >= 0.1 and sim >= self.cfg["new_cow_similarity"]
        if sure:
            state = "confirmed"
        elif sim >= self.cfg["new_cow_similarity"]:
            state = "tentative"
        else:
            state = "unknown"
        return {"state": state, "cow": best if state != "unknown" else None, "sim": sim, "margin": margin, "p": p}

    def observe(self, decision, fp, ts, track):
        """Record a decided burst: confirmed -> the cow's history; unknown -> the pool."""
        fp = _norm(fp)
        if decision["state"] == "confirmed":
            c = decision["cow"]
            self.cows[c]["last_seen"] = ts
            self.history.setdefault(c, []).append((ts, fp))
        elif decision["state"] == "unknown":
            self.pool.append({"ts": ts, "track": track, "fp": fp})

    # ------------------------------------------------------------- maintenance
    def migration_due(self, now):
        """True once per week, at the farm's change-over (default Tuesday 10:00)."""
        t = dt.datetime.fromtimestamp(now, dt.timezone(dt.timedelta(hours=self.farm["timezone_offset_hours"])))
        start = t.replace(hour=self.farm["migration_hour"], minute=0, second=0, microsecond=0)
        start -= dt.timedelta(days=(t.weekday() - self.farm["migration_weekday"]) % 7)
        if t < start:
            start -= dt.timedelta(days=7)
        ts = start.timestamp()
        if self.migration_since is None:          # first start: remember the last change-over, raise nothing
            self.migration_since = ts
            return False
        if self.migration_since < ts:
            self.migration_since = ts
            return True
        return False

    def retire(self, now):
        """After a change-over: cows not seen since it, for retire_after_hours, have left."""
        if self.migration_since is None or now - self.migration_since < self.cfg["retire_after_hours"] * 3600:
            return []
        gone = [c for c, v in self.cows.items() if v["last_seen"] < self.migration_since]
        for c in gone:
            self.cows.pop(c)
            self.history.pop(c, None)
        return gone

    def rebuild(self, now):
        """Nightly: prototypes from the last history_days of confirmed fingerprints."""
        keep_after = now - self.cfg["history_days"] * 86400
        for c in list(self.history):
            h = [(t, f) for t, f in self.history[c] if t >= keep_after][-2000:]
            self.history[c] = h
            if len(h) >= 3:
                X = np.stack([f for _, f in h])
                k = min(self.cfg["prototypes_per_cow"], max(1, len(h) // 5))
                self.cows[c]["prototypes"] = spherical_kmeans(X, k).astype(np.float32)
        self.pool = [p for p in self.pool if p["ts"] >= keep_after]

    def enroll(self, now):
        """Group the unknown pool into new cows. Bursts of one track are one cow,
        so tracks are grouped (by their mean fingerprint), greedily, then the
        groups seen often enough become cows."""
        if not self.pool:
            return []
        by_track = {}
        for p in self.pool:
            by_track.setdefault(p["track"], []).append(p)
        units = [(tr, _norm(np.mean([p["fp"] for p in ps], 0)), ps) for tr, ps in by_track.items()]
        units.sort(key=lambda u: -len(u[2]))
        groups = []                                    # [centre, [units]]
        thr = self.cfg["new_cow_similarity"]
        for u in units:
            sims = [float(g[0] @ u[1]) for g in groups]
            if sims and max(sims) >= thr:
                g = groups[int(np.argmax(sims))]
                g[1].append(u)
                g[0] = _norm(np.mean([x[1] for x in g[1]], 0))
            else:
                groups.append([u[1], [u]])
        day = dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y%m%d")
        made = []
        for centre, members in groups:
            bursts = [p for _, _, ps in members for p in ps]
            if len(bursts) < self.cfg["new_cow_min_bursts"]:
                continue
            n = self.next_new.get(day, 0) + 1
            self.next_new[day] = n
            cid = f"new-{day}-{n:02d}"
            X = np.stack([p["fp"] for p in bursts])
            k = min(self.cfg["prototypes_per_cow"], max(1, len(bursts) // 5))
            self.cows[cid] = {"prototypes": spherical_kmeans(X, k).astype(np.float32), "created": now,
                              "last_seen": max(p["ts"] for p in bursts)}
            self.history[cid] = [(p["ts"], p["fp"]) for p in bursts]
            made.append(cid)
            used = {id(p) for p in bursts}
            self.pool = [p for p in self.pool if id(p) not in used]
        return made

    def nightly(self, now):
        """retire -> rebuild -> enroll; returns what changed, for the log."""
        gone = self.retire(now)
        self.rebuild(now)
        made = self.enroll(now)
        self.save()
        return {"retired": gone, "enrolled": made, "cows": len(self.cows), "pool": len(self.pool)}
