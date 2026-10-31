#!/usr/bin/env python3
"""
tod_volume_anomaly.py -- time-of-day volume anomaly detector (median / MAD)

Two ideas, one detector
-----------------------
1. TIME-OF-DAY BASELINE
   Intraday volume is U-shaped: heavy at the open and close, light at midday.
   A volume of 80,000 is normal at 09:15 and a screaming anomaly at 12:30.
   So every bar is compared only with the SAME clock time (the "12:30 bucket")
   on the previous N trading days, never with the whole day pooled together.

2. ROBUST DISPERSION (median and MAD instead of mean and standard deviation)
   One crazy bar inflates the mean and the standard deviation, and then hides
   the next anomaly. Median and MAD barely move.

       MAD         = median(|x_i - median(x)|)
       modified z  = 0.6745 * (x_i - median(x)) / MAD

   0.6745 = 1 / 1.4826. It makes MAD comparable to a standard deviation when
   the data are roughly normal. |modified z| > 3.5 is the usual outlier cut-off
   (Iglewicz and Hoaglin, 1993).

No look-ahead: the baseline for today's bar uses only PREVIOUS days.

An extra you will want (log=True)
---------------------------------
Volume is right-skewed (a few huge bars, no negative ones), and a symmetric
cut-off on a skewed variable fires too often. Running the same maths on
log(volume) fixes most of that. The demo measures the difference. log=False
is the plain formula exactly as specified.

What is in this file
--------------------
detect()            batch version (pandas), for research on a whole history
TimeOfDayBaseline   same maths one bar at a time, for a live feed
report()            prints the results in plain language (alerts, severity, when in the day)
make_synthetic()    fake U-shaped volume with anomalies planted at known places
run_demo()          compares the methods on that data, optionally draws a chart

Usage
-----
    python tod_volume_anomaly.py                      # demo on synthetic data
    python tod_volume_anomaly.py --plot demo.png      # demo plus chart
    python tod_volume_anomaly.py --csv bars.csv       # your own data
    python tod_volume_anomaly.py --csv bars.csv --log --window 40 --min-days 20
    python tod_volume_anomaly.py --csv bars.csv --last-days 5      # only the latest 5 days

bars.csv needs two columns: timestamp, volume  (one row per bar, any bar size).
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
K = 0.6745              # = 1 / 1.4826
THRESHOLD = 3.5         # flag when modified z is above this (high side only)
WINDOW = 20             # trailing days kept for each time-of-day bucket
MIN_DAYS = 10           # no score until a bucket has this many days of history
MAD_FLOOR_FRAC = 0.05   # MAD may never fall below 5% of the median ...
MAD_FLOOR_ABS = 1.0     # ... or below 1 share. Stops z exploding when the last
                        # N days of a bucket happen to be almost identical.
LOG_MAD_FLOOR = 0.02    # the same safety net when working in log units


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def _mad(x) -> float:
    """MAD = median(|x_i - median(x)|). NaNs (missing bars) are ignored."""
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(np.abs(x - np.nanmedian(x))))


def _to_log(v):
    return np.log(np.maximum(v, 1.0))       # a volume of 0 becomes log(1) = 0


def to_matrix(volume: pd.Series) -> pd.DataFrame:
    """Reshape to rows = trading days, columns = clock time ('09:15', '09:20', ...).

    Rolling DOWN a column then means "the same clock time on previous days".
    """
    idx = volume.index
    tidy = pd.DataFrame({
        "date": idx.normalize(),
        "tod": idx.strftime("%H:%M"),
        "vol": volume.to_numpy(),
    })
    # pivot raises on duplicate (date, time) rows, which is what we want
    return tidy.pivot(index="date", columns="tod", values="vol").sort_index()


def baseline(mat: pd.DataFrame, window=WINDOW, min_days=MIN_DAYS,
             floor_frac=MAD_FLOOR_FRAC, floor_abs=MAD_FLOOR_ABS):
    """Time-matched rolling median and MAD (two matrices shaped like `mat`)."""
    hist = mat.shift(1)                 # shift(1): today is NOT in its own baseline
    roll = hist.rolling(window, min_periods=min_days)
    med = roll.median()
    mad = roll.apply(_mad, raw=True)
    floor = np.maximum(floor_frac * med, floor_abs)
    mad = mad.where(mad >= floor, floor)
    return med, mad


def modified_z(x, med, mad):
    return K * (x - med) / mad


# ----------------------------------------------------------------------------
# Batch detector
# ----------------------------------------------------------------------------
def detect(volume: pd.Series, window=WINDOW, min_days=MIN_DAYS, threshold=THRESHOLD,
           two_sided=False, log=False, floor_frac=MAD_FLOOR_FRAC, floor_abs=MAD_FLOOR_ABS):
    """Score every bar. Returns a table with the same index as `volume`:

        volume | median (normal for this clock time) | mad | z (modified z)
               | upper (volume above which the bar is flagged) | flag

    With log=True the maths runs on log(volume); `mad` is then in log units,
    while `median` and `upper` are converted back to volume.
    """
    if log:
        values, floor_frac, floor_abs = _to_log(volume), 0.0, LOG_MAD_FLOOR
    else:
        values = volume
    mat = to_matrix(values)
    med, mad = baseline(mat, window, min_days, floor_frac, floor_abs)
    z = modified_z(mat, med, mad)
    upper = med + threshold * mad / K       # z = threshold  <=>  x = upper
    if log:
        med, upper = np.exp(med), np.exp(upper)

    # look each bar's (day, clock time) up in the matrices
    r = mat.index.get_indexer(volume.index.normalize())
    c = mat.columns.get_indexer(volume.index.strftime("%H:%M"))
    out = pd.DataFrame({
        "volume": volume.to_numpy(),
        "median": med.to_numpy()[r, c],
        "mad": mad.to_numpy()[r, c],
        "z": z.to_numpy()[r, c],
        "upper": upper.to_numpy()[r, c],
    }, index=volume.index)
    score = out["z"].abs() if two_sided else out["z"]
    out["flag"] = score > threshold     # NaN (still warming up) compares False
    return out


# ----------------------------------------------------------------------------
# Live detector: same maths, one bar at a time
# ----------------------------------------------------------------------------
class TimeOfDayBaseline:
    """Keeps the last `window` values for each clock-time bucket.

        det = TimeOfDayBaseline()
        det.fit(history_series)                 # optional warm-up
        for ts, vol in live_bars:
            result = det.on_bar(ts, vol)        # score first, then learn
            if result["flag"]: ...alert...

    Note: an anomalous bar is still added to its bucket afterwards. That is
    fine, because the median and MAD hardly notice one outlier in 20 values.
    A missing bar simply is not stored (the batch version keeps an empty slot,
    so the two differ only when bars are missing).
    """

    def __init__(self, window=WINDOW, min_days=MIN_DAYS, threshold=THRESHOLD,
                 log=False, floor_frac=MAD_FLOOR_FRAC, floor_abs=MAD_FLOOR_ABS):
        self.min_days = min_days
        self.threshold = threshold
        self.log = log
        self.floor_frac, self.floor_abs = (0.0, LOG_MAD_FLOOR) if log else (floor_frac, floor_abs)
        self._hist = defaultdict(lambda: deque(maxlen=window))

    @staticmethod
    def bucket(ts) -> str:
        return ts if isinstance(ts, str) else pd.Timestamp(ts).strftime("%H:%M")

    def score(self, ts, volume) -> dict:
        b = self.bucket(ts)
        h = self._hist[b]
        if len(h) < self.min_days:
            return {"bucket": b, "ready": False, "median": np.nan, "mad": np.nan,
                    "z": np.nan, "upper": np.nan, "flag": False}
        x = np.fromiter(h, dtype=float)
        med = float(np.median(x))
        mad = max(_mad(x), self.floor_frac * med, self.floor_abs)
        v = float(_to_log(volume)) if self.log else float(volume)
        z = K * (v - med) / mad
        upper = med + self.threshold * mad / K
        if self.log:
            med, upper = float(np.exp(med)), float(np.exp(upper))
        return {"bucket": b, "ready": True, "median": med, "mad": mad, "z": z,
                "upper": upper, "flag": bool(z > self.threshold)}

    def update(self, ts, volume) -> None:
        if np.isfinite(volume):
            self._hist[self.bucket(ts)].append(float(_to_log(volume)) if self.log else float(volume))

    def on_bar(self, ts, volume) -> dict:
        result = self.score(ts, volume)     # uses history only
        self.update(ts, volume)
        return result

    def fit(self, volume: pd.Series) -> None:
        for ts, v in volume.items():
            self.update(ts, v)


# ----------------------------------------------------------------------------
# For the comparison only: the two weaker methods
# ----------------------------------------------------------------------------
def z_tod_mean_std(mat, window=WINDOW, min_days=MIN_DAYS):
    """Time-of-day baseline, but with ordinary mean and standard deviation."""
    roll = mat.shift(1).rolling(window, min_periods=min_days)
    return (mat - roll.mean()) / roll.std()


def z_pooled_mean_std(mat, window=WINDOW, min_days=MIN_DAYS):
    """The naive way: ONE mean and std over every bar of the last `window` days."""
    z = np.full(mat.shape, np.nan)
    for i in range(min_days, len(mat)):
        past = mat.iloc[max(0, i - window):i].to_numpy().ravel()
        z[i] = (mat.iloc[i].to_numpy() - np.nanmean(past)) / np.nanstd(past, ddof=1)
    return pd.DataFrame(z, index=mat.index, columns=mat.columns)


# ----------------------------------------------------------------------------
# Synthetic data with known answers
# ----------------------------------------------------------------------------
BARS = list(pd.date_range("2000-01-01 09:15", "2000-01-01 15:25", freq="5min").strftime("%H:%M"))

# (day number, clock time, multiplier, why it is in the test)
PLANTED = [
    (100, "12:30", 40.0, "huge midday spike; sits in the 12:30 baseline for the next 20 days"),
    (117, "12:30", 3.5, "midday x3.5: lower than a normal OPEN bar, still abnormal for 12:30"),
    (118, "10:00", 3.0, "morning x3"),
    (119, "11:20", 4.0, "late-morning x4"),
    (119, "15:05", 4.0, "near-close x4"),
]


def make_synthetic(n_days=120, seed=7, mid_volume=20_000):
    """5-minute bars, 09:15-15:25 (75 per day), U-shaped, with PLANTED anomalies."""
    rng = np.random.default_rng(seed)
    nb = len(BARS)
    x = np.linspace(-1, 1, nb)
    shape = 1 + 3 * x ** 2                              # open/close = 4x midday
    day_level = rng.lognormal(0, 0.15, n_days)          # busy day / quiet day
    noise = rng.lognormal(0, 0.25, (n_days, nb))        # bar-to-bar randomness
    vol = mid_volume * shape[None, :] * day_level[:, None] * noise

    days = pd.bdate_range(end="2026-10-30", periods=n_days)
    for d, t, mult, _ in PLANTED:
        vol[d, BARS.index(t)] *= mult

    offsets = pd.to_timedelta([b + ":00" for b in BARS]).to_numpy()
    idx = days.repeat(nb) + np.tile(offsets, n_days)
    series = pd.Series(vol.round().ravel(), index=idx, name="volume")
    planted_ts = [(days[d] + pd.Timedelta(t + ":00"), why) for d, t, _, why in PLANTED]
    return series, planted_ts


# ----------------------------------------------------------------------------
# Reading the results in plain language
# ----------------------------------------------------------------------------
LEVELS = [(10.0, "EXTREME"), (5.0, "HIGH"), (THRESHOLD, "WATCH")]   # my own tiers, change freely


def _level(z: float) -> str:
    for cut, name in LEVELS:
        if z >= cut:
            return name
    return ""


def report(res: pd.DataFrame, last_days=None, top=15, notes=None, title=None, legend=True):
    """Print detect() output in plain language.

    Shows how much was flagged, WHEN in the day the flags fall, and a table of
    the strongest alerts: volume, what is normal for that clock time, how many
    times normal it was, and a severity level.

    last_days : only list alerts from the most recent N days
    notes     : optional {timestamp: text} shown in a NOTE column
    """
    notes = notes or {}
    scored = res["z"].notna()
    flagged = res[res["flag"]]
    n_scored, n_flag = int(scored.sum()), len(flagged)
    if title:
        print(title)
    print(f"   Whole history: checked {n_scored:,} bars over {res.index[scored].normalize().nunique()} days "
          f"and flagged {n_flag} ({n_flag / max(n_scored, 1):.2%}).")

    # when in the day do the flags fall? (a good baseline should not pile up at open/close)
    tods = sorted(res.index.strftime("%H:%M").unique())
    k = max(1, round(len(tods) / 6))
    pos = pd.Series(np.arange(len(tods)), index=tods)

    def part_of_day(index):
        p = pos.reindex(index.strftime("%H:%M")).to_numpy()
        return pd.Series(np.where(p < k, "open", np.where(p >= len(tods) - k, "close", "midday")))

    all_part, flag_part = part_of_day(res.index[scored]), part_of_day(flagged.index)
    cells = [f"{name} {int((flag_part == name).sum())} flags ({(all_part == name).mean():.0%} of bars)"
             for name in ("open", "midday", "close")]
    print("   Where in the day (whole history):  " + "  |  ".join(cells))

    per_day = flagged.groupby(flagged.index.normalize()).size().sort_values(ascending=False)
    busy = per_day[per_day >= 4].head(3)
    if len(busy):
        print("   Days with 4+ flags (whole history): " + ", ".join(f"{d:%d %b} ({n})" for d, n in busy.items())
              + "  (a cluster usually means a busy market day, not one odd bar)")

    show = flagged
    if last_days:
        show = flagged[flagged.index >= res.index.normalize().unique()[-last_days]]
    if show.empty:
        print("\n   No bars flagged" + (f" in the last {last_days} days." if last_days else "."))
    else:
        strongest = show.nlargest(top, "z").sort_index()
        scope = f"in the last {last_days} days" if last_days else "in total"
        more = f"  (strongest {len(strongest)} of {len(show)})" if len(show) > top else ""
        print(f"\n   ALERTS {scope}{more}")
        head = f"   {'WHEN':<13}{'VOLUME':>10}{'NORMAL':>10}{'LIMIT':>10}{'x NORMAL':>10}{'SCORE':>8}  {'LEVEL':<9}"
        print(head + ("NOTE" if notes else ""))
        for ts, r in strongest.iterrows():
            print(f"   {ts:%d %b %H:%M}  {r['volume']:>10,.0f}{r['median']:>10,.0f}{r['upper']:>10,.0f}"
                  f"{r['volume'] / r['median']:>9.1f}x{r['z']:>8.1f}  {_level(r['z']):<9}"
                  f"{notes.get(ts, '')}")
    if legend:
        print("\n   How to read it")
        print("     NORMAL    typical volume at this clock time (median of the recent days)")
        print("     LIMIT     volume above which a bar gets flagged")
        print("     x NORMAL  volume divided by normal")
        print(f"     SCORE     modified z-score ({THRESHOLD} is the flag line; higher = stranger)")
        print("     LEVEL     " + "  ".join(f"{n} {c:g}+" for c, n in reversed(LEVELS))
              + "   (my own tiers; change LEVELS)")


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
BURN_IN = 40    # days skipped when counting, so every variant has a full window


def run_demo(plot_path=None):
    vol, planted = make_synthetic()
    mat = to_matrix(vol)
    n_days, n_bars = mat.shape
    bar = "=" * 80
    print(f"{bar}\nTIME-OF-DAY VOLUME ANOMALY DETECTOR  -  demo on made-up data\n{bar}")
    print(f"{n_days} trading days x {n_bars} five-minute bars. {len(planted)} anomalies are hidden in it,")
    print("so we know the right answer and can score each method.\n")

    def zmat(**kw):
        return to_matrix(detect(vol, **kw)["z"])

    variants = {
        "flat":  ("Flat limit (one mean/SD for the whole day)", z_pooled_mean_std(mat)),
        "avg":   ("Same-time mean/SD",                          z_tod_mean_std(mat)),
        "mad":   ("Same-time median/MAD (plain formula)",           zmat()),
        "mad40": ("Same-time median/MAD, 40-day window",        zmat(window=40, min_days=20)),
        "log":   ("Log volume + median/MAD",                    zmat(log=True)),
        "log40": ("Log volume + median/MAD, 40-day window",     zmat(window=40, min_days=20, log=True)),
    }

    planted_mask = pd.DataFrame(False, index=mat.index, columns=mat.columns)
    for ts, _ in planted:
        planted_mask.loc[ts.normalize(), ts.strftime("%H:%M")] = True
    scored_rows = pd.Series(np.arange(n_days) >= BURN_IN, index=mat.index)
    n_eval_days = int(scored_rows.sum())

    # ---- 1) scoreboard ------------------------------------------------
    print(f"1) SCOREBOARD   (days {BURN_IN + 1}-{n_days}; flagged = score above {THRESHOLD})")
    print(f"   {'METHOD':<46}{'CAUGHT':>10}{'FALSE ALARMS':>15}{'PER DAY':>9}")
    flags, stats = {}, {}
    for key, (label, zm) in variants.items():
        f = (zm > THRESHOLD) & zm.notna() & scored_rows.to_numpy()[:, None]
        flags[key] = f
        caught = int((f & planted_mask).sum().sum())
        false = int((f & ~planted_mask).sum().sum())
        stats[key] = (caught, false)
        print(f"   {label:<46}{caught:>7} of {len(planted)}{false:>15}{false / n_eval_days:>9.2f}")
    top_caught = max(c for c, _ in stats.values())
    best = min((k for k, (c, _) in stats.items() if c == top_caught), key=lambda k: stats[k][1])
    print(f"   -> Best on this data: {variants[best][0]}  "
          f"[{stats[best][0]} of {len(planted)} caught, {stats[best][1] / n_eval_days:.2f} false alarms a day]")
    print("      (a false alarm = a flag on a bar where nothing was hidden)\n")

    # ---- 2) the hidden anomalies, one by one --------------------------
    med = to_matrix(detect(vol)["median"])
    cols = [("flat", "Flat limit", 12), ("avg", "Same-time avg", 15), ("mad", "Median/MAD", 12), ("log", "Log", 8)]
    print("2) THE HIDDEN ANOMALIES   (who caught each one; 20-day windows)")
    print(f"   {'WHEN':<13}{'VOLUME':>10}{'NORMAL':>10}{'x NORMAL':>10}   "
          + "".join(f"{name:^{w}}" for _, name, w in cols))
    for ts, why in planted:
        dt, t = ts.normalize(), ts.strftime("%H:%M")
        v, norm = mat.loc[dt, t], med.loc[dt, t]
        marks = "".join(f"{('yes' if flags[k].loc[dt, t] else 'MISSED'):^{w}}" for k, _, w in cols)
        print(f"   {ts:%d %b %H:%M}  {v:>10,.0f}{norm:>10,.0f}{v / norm:>9.1f}x   {marks}")
        print(f"      what it tests: {why}")
    print()

    # ---- 3) what an analyst would actually see ------------------------
    planted_ts = {ts for ts, _ in planted}
    res = detect(vol)
    notes = {ts: ("HIDDEN anomaly" if ts in planted_ts else "false alarm") for ts in res.index[res["flag"]]}
    report(res, last_days=3, notes=notes,
           title="3) WHAT AN ANALYST WOULD SEE ON THE LAST 3 DAYS   (plain formula, 20-day window)")
    res40 = detect(vol, window=40, min_days=20, log=True)
    cutoff = res.index.normalize().unique()[-3]
    last3 = res40[res40["flag"] & (res40.index >= cutoff)]
    real = sum(ts in planted_ts for ts in last3.index)
    print(f"\n   Same 3 days with log volume + 40-day window: {len(last3)} flags, "
          f"{real} hidden anomalies and {len(last3) - real} false alarms.\n")

    # ---- 4) self-check -------------------------------------------------
    print("4) SELF-CHECK   (the batch code and the live one-bar-at-a-time code must agree)")
    for log in (False, True):
        batch = detect(vol, log=log)
        live = TimeOfDayBaseline(log=log)
        live_z = np.array([live.on_bar(ts, v)["z"] for ts, v in vol.items()])
        same = np.allclose(batch["z"].to_numpy(), live_z, equal_nan=True)
        print(f"   {'log volume' if log else 'plain volume'}: {'OK' if same else 'MISMATCH'} "
              f"on all {vol.size:,} bars")

    if plot_path:
        plot_demo(vol, planted, plot_path)
        print(f"\nChart saved: {plot_path}")


def plot_demo(vol, planted, path):
    """Two panels. Left: why a time-of-day limit. Right: why median/MAD."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    GRID, AXIS = "#e1e0d9", "#c3c2b7"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    kfmt = FuncFormatter(lambda v, _: f"{v / 1000:,.0f}K" if v else "0")

    res = detect(vol)
    mat = to_matrix(vol)
    upper = to_matrix(res["upper"])
    flag = to_matrix(res["flag"].astype(float)) > 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=150, facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        ax.yaxis.set_major_formatter(kfmt)

    # ---- left: the last day, bar by bar --------------------------------
    last = mat.index[-1]
    x = np.arange(len(BARS))
    today = mat.loc[last].to_numpy()
    lim_c = upper.loc[last].to_numpy()
    past = mat.iloc[-1 - WINDOW:-1].to_numpy().ravel()
    lim_a = np.nanmean(past) + THRESHOLD * np.nanstd(past, ddof=1)
    fl = flag.loc[last].to_numpy()

    ax1.fill_between(x, 0, lim_c, color=ORANGE, alpha=0.10, lw=0)
    ax1.plot(x, lim_c, color=ORANGE, lw=2)
    ax1.axhline(lim_a, color=MUTED, lw=2)
    ax1.plot(x, today, color=BLUE, lw=2)
    ax1.scatter(x[fl], today[fl], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=5)
    for ts, _ in planted:
        if ts.normalize() == last:
            i = BARS.index(ts.strftime("%H:%M"))
            right = i > len(BARS) * 0.75
            ax1.annotate(f"{ts.strftime('%H:%M')}  {today[i] / 1000:,.0f}K", (i, today[i]),
                         xytext=(-8 if right else 8, 8), textcoords="offset points",
                         ha="right" if right else "left", fontsize=9.5, color=INK2)
    planted_ts = {ts for ts, _ in planted}
    for i in np.flatnonzero(fl):                 # flagged bars that were NOT planted
        if last + pd.Timedelta(BARS[i] + ":00") not in planted_ts:
            ax1.annotate(f"{BARS[i]}  false alarm", (i, today[i]), xytext=(-4, -34),
                         textcoords="offset points", ha="right", va="top", fontsize=9, color=INK2,
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=6))
    hours =["10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]
    ax1.set_xticks([BARS.index(h) for h in hours], hours)
    ax1.set_ylim(0, max(np.nanmax(today), lim_a, np.nanmax(lim_c)) * 1.12)
    ax1.set_xlim(-1, len(BARS))
    ax1.set_ylabel("Volume per 5-minute bar", color=INK2, fontsize=10)
    ax1.set_title(f"One flat limit vs a time-of-day limit  ({last:%d %b})",
                  loc="left", fontsize=12.5, color=INK, fontweight="bold", pad=26)
    ax1.text(0, 1.03, "The flat limit sits above every midday bar, so a midday spike hides under it.",
             transform=ax1.transAxes, fontsize=9.5, color=INK2)
    ax1.legend(handles=[
        Line2D([], [], color=BLUE, lw=2, label="Volume today"),
        Line2D([], [], color=ORANGE, lw=2, label="Flag limit for this clock time (median + MAD)"),
        Line2D([], [], color=MUTED, lw=2, label="One flat limit (mean + 3.5 SD of all bars, last 20 days)"),
        Line2D([], [], color=BLUE, lw=0, marker="o", markersize=8, label="Flagged by the time-of-day limit"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)

    # ---- right: one clock time (12:30) across the last 40 days -----------
    col = "12:30"
    n = 40
    days = mat.index[-n:]
    v = mat[col].iloc[-n:].to_numpy()
    lim_c2 = upper[col].iloc[-n:].to_numpy()
    roll = mat.shift(1).rolling(WINDOW, min_periods=MIN_DAYS)
    lim_b = (roll.mean() + THRESHOLD * roll.std())[col].iloc[-n:].to_numpy()
    fl2 = flag[col].iloc[-n:].to_numpy()
    xs = np.arange(n)
    ymax = 130_000

    ax2.plot(xs, lim_b, color=MUTED, lw=2)
    ax2.fill_between(xs, 0, lim_c2, color=ORANGE, alpha=0.10, lw=0)
    ax2.plot(xs, lim_c2, color=ORANGE, lw=2)
    ax2.plot(xs, v, color=BLUE, lw=2)
    ax2.scatter(xs[fl2], v[fl2], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=5)
    spike = int(np.nanargmax(v))
    ax2.annotate(f"{v[spike] / 1000:,.0f}K (off the scale)", (spike, ymax),
                 xytext=(-18, -4), textcoords="offset points", ha="right", va="top",
                 fontsize=9.5, color=INK2)
    after = lim_b[spike + 1:]
    ax2.annotate(f"After the spike, the mean + 3.5 SD\n"
                 f"limit jumps to ~{np.nanmax(after) / 1000:,.0f}K and stays\n"
                 f"off the scale for 20 days",
                 (n - 1, ymax * 0.93), ha="right", va="top", fontsize=9.5, color=INK2)
    for ts, _ in planted:
        if ts.strftime("%H:%M") == col and ts.normalize() in set(days) and ts.normalize() != days[spike]:
            i = list(days).index(ts.normalize())
            ax2.annotate(f"{ts.strftime('%d %b')}  {v[i] / 1000:,.0f}K\nflagged only by median/MAD",
                         (i, v[i]), xytext=(-10, 14), textcoords="offset points",
                         ha="right", fontsize=9.5, color=INK2)
    tick_i = list(range(0, n, 8)) + [n - 1]
    ax2.set_xticks(tick_i, [days[i].strftime("%d %b") for i in tick_i])
    ax2.set_ylim(0, ymax)
    ax2.set_xlim(-1, n)
    ax2.set_ylabel("Volume of the 12:30 bar", color=INK2, fontsize=10)
    ax2.set_title(f"One big spike ruins a mean/SD limit  (the {col} bar, last {n} days)",
                  loc="left", fontsize=12.5, color=INK, fontweight="bold", pad=26)
    ax2.text(0, 1.03, "Median and MAD ignore the spike, so the next real anomaly is still caught.",
             transform=ax2.transAxes, fontsize=9.5, color=INK2)
    ax2.legend(handles=[
        Line2D([], [], color=BLUE, lw=2, label="Volume at 12:30"),
        Line2D([], [], color=ORANGE, lw=2, label="Flag limit (median + MAD, last 20 days)"),
        Line2D([], [], color=MUTED, lw=2, label="Flag limit (mean + 3.5 SD, last 20 days)"),
        Line2D([], [], color=BLUE, lw=0, marker="o", markersize=8, label="Flagged by median/MAD"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.2, wspace=0.14)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Your own data
# ----------------------------------------------------------------------------
def run_csv(path, window, min_days, threshold, log, out=None, last_days=None):
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    res = detect(df["volume"], window=window, min_days=min_days, threshold=threshold, log=log)
    mode = "log volume" if log else "plain volume"
    report(res, last_days=last_days, top=20,
           title=f"VOLUME ANOMALIES in {path}   ({mode}, {window}-day window, flag above {threshold})")
    if out:
        res.to_csv(out)
        print(f"\n   Full scored table saved to {out}")


def main():
    ap = argparse.ArgumentParser(description="Time-of-day median/MAD volume anomaly detector")
    ap.add_argument("--csv", help="CSV with columns timestamp,volume")
    ap.add_argument("--out", help="with --csv: save the scored table here")
    ap.add_argument("--plot", help="with the demo: save the chart to this PNG")
    ap.add_argument("--window", type=int, default=WINDOW, help="days of history per clock time")
    ap.add_argument("--min-days", type=int, default=MIN_DAYS)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--log", action="store_true", help="run the maths on log(volume)")
    ap.add_argument("--last-days", type=int, help="with --csv: only list alerts from the last N days")
    a = ap.parse_args()
    if a.csv:
        run_csv(a.csv, a.window, a.min_days, a.threshold, a.log, a.out, a.last_days)
    else:
        run_demo(a.plot)


if __name__ == "__main__":
    main()
