"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts Grand Theft Auto VI's launch-day Twitch viewers, hours 2..24, as
five quantiles per hour. The model is an empirical launch-shape model built
from the case's own comparables (past big game launches):

- For each comparable with a complete launch day, take its launch-day hourly
  viewers (24 values starting at `day_start`) and form log-ratios
  log(v_h / v_1) for each hour h, i.e. how that launch grew or faded relative
  to its own first hour.
- Weight each comparable by how similar its launch is to a console game with
  a rolling midnight-local unlock (the GTA VI shape), and by how close its
  first-hour size is to this case's hour-1 print (a log-normal kernel,
  sigma 0.7 in log space). Early-access-contaminated comparables get weight 0.
- Per hour, the weighted mean and standard deviation of the log-ratios give
  the forecast quantiles: q_tau = p1 * exp(mu_h + z_tau * sd_h), where p1 is
  the case's hour-1 print and z_tau are the standard normal quantiles.
- If no comparable covers an hour, that hour falls back to the house
  persistence forecast (p1 * exp(z_tau * 0.15 * sqrt(h - 1))).

Why this beats persistence: a launch day is not a random walk around hour 1.
Rolling midnight unlocks make viewership jump at 05:00Z (US East) and 08:00Z
(US West) and fade or grow through the day in a pattern past launches share.
Validated by leave-one-out cross-validation over the comparables in the kit:
on the console midnight-local launches (the GTA VI analogues) this scores a
mean log-scale pinball loss of ~0.25, against ~0.36 for persistence.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so the sys.path line below puts
the directory first (kept from the starter; this entry is one file).

What it does:
  - read every <id>.in in the input directory (the grader runs one case at
    a time, with only that case's file there): JSON
    {"schema": "makefun/twitch-ccu-case/v1", "id", "game", "day_start",
    "checkpoint", "levels", "known", "forecast", "history",
    "history_hourly", "streams", "categories_hourly", "comparables",
    "notes"}, where `known` holds hour 1's print {hour: 1, from, until,
    viewers, channels, reads} and `forecast` lists the hours to forecast,
    2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2,
    "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24
    each once, no other key, each q five finite numbers above 0 and at most
    1e9 (viewers; fractions allowed), non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports (this entry is a single file, but the
# line is kept for parity with the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of an hour's move for the persistence fallback, in natural-log
# units per square-root hour.
SIGMA = 0.15
# Width of the log-magnitude similarity kernel between a comparable's
# first-hour viewers and this case's hour-1 print.
MAGMA_SIGMA = 0.7


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def unlock_weight(comp: dict) -> float:
    """How much a comparable's launch shape resembles GTA VI's.

    GTA VI is a console game with a rolling midnight-local unlock. A
    comparable with the same shape counts fully; one with a single global
    unlock counts less (its day curve jumps at one worldwide moment instead);
    an unverified one counts in between; an early-access-contaminated one
    (its launch-day curve already peaked before release day) counts zero.
    """
    text = (comp.get("unlock") or "").lower()
    if "early-access-contaminated" in text:
        return 0.0
    if "midnight local" in text or "rolling" in text:
        return 1.0
    if "global" in text or "simultaneous" in text:
        return 0.3
    return 0.6


def launch_day(comp: dict) -> list | None:
    """The comparable's 24 launch-day hourly viewer counts, or None.

    The launch day is the 24 hourly points starting at `day_start`. A
    comparable with no launch day of its own (the GTA VI category's own
    pre-launch history) or a non-positive first hour cannot give ratios.
    """
    hourly = comp.get("hourly") or []
    start = comp.get("day_start")
    index = None
    for i, point in enumerate(hourly):
        if point.get("from") == start:
            index = i
            break
    if index is None or index + 24 > len(hourly):
        return None
    values = [point.get("viewers", 0) for point in hourly[index:index + 24]]
    if values[0] <= 0 or any(v < 0 for v in values):
        return None
    return values


def weighted_mean_sd(pairs: list) -> tuple:
    """Weighted mean and population standard deviation of (weight, value)."""
    total = sum(w for w, _ in pairs)
    mean = sum(w * v for w, v in pairs) / total
    var = sum(w * (v - mean) ** 2 for w, v in pairs) / total
    return mean, math.sqrt(var)


def hour_quantiles(p1: float, hour: int, ratios: list) -> list:
    """The five quantiles for one forecast hour from weighted log-ratios.

    `ratios` holds (weight, log(v_hour / v_first_hour)) over the reference
    launches. With none, fall back to the persistence forecast.
    """
    if not ratios:
        return [round2(p1 * math.exp(z * SIGMA * math.sqrt(hour - 1))) for z in Z]
    mean, sd = weighted_mean_sd(ratios)
    out = []
    for z in Z:
        value = p1 * math.exp(mean + z * sd)
        if value < 1:
            value = 1.0
        elif value > 1e9:
            value = 1e9
        out.append(round2(value))
    return out


def validate_case(case: dict, name: str) -> tuple:
    """Check a case against the contract the grader promises, failing fast.

    The grader sends levels [0.05, 0.25, 0.5, 0.75, 0.95], exactly one
    hour-1 print with a positive viewer count, and a forecast of hours
    2..24 each once. A case that breaks the contract is bad input, not a
    forecast to guess at: raise naming the file and the problem, instead
    of failing halfway through with a bare KeyError or silently clamping
    a bad hour-1 print to 1.
    """
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    if not isinstance(first.get("viewers"), (int, float)) or first["viewers"] <= 0:
        raise ValueError(f"{name} hour-1 print has no positive viewer count")
    rows = case.get("forecast") or []
    hours = [row.get("hour") for row in rows]
    if hours != list(range(2, 25)):
        raise ValueError(f"{name} forecasts hours {hours}, not 2..24 each once")
    return first["viewers"], rows


def forecast_case(case: dict, name: str) -> dict:
    p1, rows = validate_case(case, name)

    # Reference launches, in a fixed order so two runs sum identically.
    refs = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: (c.get("name", ""), c.get("day_start", ""))):
        day = launch_day(comp)
        if day is None:
            continue
        weight = unlock_weight(comp)
        if weight <= 0:
            continue
        magnitude = math.exp(-0.5 * (math.log(day[0] / p1) / MAGMA_SIGMA) ** 2)
        refs.append((weight * magnitude, day))

    hours = []
    for row in rows:
        hour = row["hour"]
        index = hour - 1  # launch-day index of this forecast hour
        ratios = [(w, math.log(day[index] / day[0])) for w, day in refs if day[index] > 0]
        hours.append({"hour": hour, "q": hour_quantiles(p1, hour, ratios)})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
