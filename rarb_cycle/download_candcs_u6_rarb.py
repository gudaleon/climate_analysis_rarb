#!/usr/bin/env python3
"""
CanDCS-U6 (PCIC) single-cell station extraction, per Prompt_for_Alex_PCIC_download.md.

Rules honoured:
- Dates ONLY from each file's time variable, decoded with its own calendar
  attribute via cftime.num2date. Dates are never synthesised by counting days.
- No value is changed, filled, smoothed or interpolated. Unit conversion only
  where the file's own `units` attribute requires it, and it is recorded.
- Anything unexpected is logged to ISSUES.md, never silently fixed.
- Comparison with existing thesis files (spec section 6) is NOT done here:
  Alex confirmed 2026-10-19 that Jeremy will compare.
Outputs per run folder:
  raw_nc/{STATION}_{MODEL}_{var}.nc   full-precision raw subsets, time+calendar kept
  txt/{STATION}_{MODEL}.txt          tab-separated, no header, 1 decimal, -99.9 missing
  metadata.csv                       one row per station x model x variable
  checks_report.md                   section-5 checks, reported not fixed
  download_log.txt, ISSUES.md, code/download_pcic.py (copy)
"""
import argparse, csv, hashlib, json, math, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cftime
import xarray as xr

CATALOG_URL = "https://services.pacificclimate.org/data/downscaled_cmip6/catalog.json"
OPENDAP_BASE = "https://services.pacificclimate.org/data/downscaled_cmip6/"
MODELS = ["ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5", "CNRM-ESM2-1", "EC-Earth3-Veg",
          "GFDL-ESM4", "INM-CM5-0", "MIROC-ES2L", "MRI-ESM2-0", "NorESM2-LM"]
VARS = ["pr", "tasmax", "tasmin"]
STATIONS = {  # label: (lat, lon) -- RARB adaptation, logged in ASSUMPTIONS_OF_RECORD.md
    "WINNIPEG": (49.92, -97.23),
    "BRANDON": (49.91, -99.95),
    "PORTAGE": (49.96, -98.28),
    "EMERSON": (49.03, -97.18),
    "MORRIS": (49.35, -97.37),
    # One-step north-east neighbour cells: the target coordinate is the primary
    # cell centre plus one grid spacing (1/12 degree) in latitude and longitude,
    # read from the ACCESS-ESM1-5 file's own grid during acquisition.
    "WINNIPEGNE": (50.041666, -97.125002),
    "BRANDONNE": (49.958333, -99.875002),
    "PORTAGENE": (50.041666, -98.208335),
    "EMERSONNE": (49.125000, -97.125002),
    "MORRISNE": (49.458333, -97.291668),
}
EXPECTED_ROWS = {"365_day": 55115, "noleap": 55115}
STANDARD_CALS = {"standard", "gregorian", "proleptic_gregorian"}

log_lines, issues = [], []

def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)

def issue(msg):
    log(f"ISSUE: {msg}")
    issues.append(msg)

def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))

def conversion_for(var, units):
    """Return (factor, offset, out_units, description) based ONLY on the file's units attr."""
    u = (units or "").strip()
    if var == "pr":
        if u == "kg m-2 s-1":
            return 86400.0, 0.0, "mm/day", "pr: kg m-2 s-1 -> mm/day (x86400)"
        if u in ("kg m-2 d-1", "kg m-2 day-1", "mm/day", "mm d-1"):
            return 1.0, 0.0, "mm/day", f"pr: units already {u} (= mm/day), no conversion"
        return None
    else:  # tasmax / tasmin
        if u in ("K", "Kelvin"):
            return 1.0, -273.15, "degC", f"{var}: K -> degC (-273.15)"
        if u in ("degC", "degrees_C", "Celsius", "deg_C"):
            return 1.0, 0.0, "degC", f"{var}: units already {u}, no conversion"
        return None

def _eq(a, b):
    return (a == b) or (np.isnan(a) and np.isnan(b))

def max_runs_and_long_runs(vals_rounded, dates):
    """Return ((value, longest_len), list of (start_date, value, length) for runs >= 10)."""
    long_runs, best = [], (None, 0)
    n = len(vals_rounded)
    if n == 0:
        return best, long_runs
    start = 0
    for i in range(1, n + 1):
        if i == n or not _eq(vals_rounded[i], vals_rounded[start]):
            length = i - start
            cur = vals_rounded[start]
            if length > best[1]:
                best = (cur, length)
            if length >= 10 and not np.isnan(cur):
                d = dates[start]
                long_runs.append((f"{d.year:04d}-{d.month:02d}-{d.day:02d}", float(cur), length))
            start = i
    return best, long_runs

def summer_peak_doy(dates, tmax):
    """DOY of peak of 31-day running mean of DOY-mean tmax, per spec section 5.
    DOY axis = ordered (month, day) sequence of the model's own calendar."""
    out = {}
    for label, (y0, y1) in {"1951-1980": (1951, 1980), "2071-2100": (2071, 2100)}.items():
        md_keys, sums, counts = [], {}, {}
        for d, v in zip(dates, tmax):
            if y0 <= d.year <= y1 and not np.isnan(v):
                k = (d.month, d.day)
                if k not in sums:
                    sums[k], counts[k] = 0.0, 0; md_keys.append(k)
                sums[k] += v; counts[k] += 1
        md_keys.sort()
        means = np.array([sums[k]/counts[k] for k in md_keys])
        if len(means) >= 31:
            kernel = np.ones(31)/31
            smooth = np.convolve(means, kernel, mode="same")
            # edges use partial windows via convolve 'same' + renormalisation
            w = np.convolve(np.ones_like(means), kernel, mode="same")
            smooth = smooth/np.where(w > 0, w, 1)
        else:
            smooth = means
        idx = int(np.nanargmax(smooth))
        out[label] = (idx + 1, md_keys[idx])  # 1-based sequential DOY index + (month, day)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out = Path(args.out); (out/"raw_nc").mkdir(parents=True, exist_ok=True); (out/"txt").mkdir(parents=True, exist_ok=True)
    log(f"Run start. Models: {models}")

    with urllib.request.urlopen(CATALOG_URL, timeout=120) as r:
        catalog = json.load(r)
    log(f"Catalogue entries: {len(catalog)}")

    meta_rows, check_lines = [], []
    check_lines.append("# Section-5 checks (reported, not fixed)\n")

    for model in models:
        # per-model storage: data[station][var] = dict(dates, vals_out, vals_raw, meta...)
        data = {s: {} for s in STATIONS}
        calendars = {}
        for var in VARS:
            # exact catalogue URL for this model/var/ssp585 (member + grid label from PCIC, never guessed)
            hits = [(k, v) for k, v in catalog.items()
                    if model in k and var == k.split("_")[0] and "ssp585" in k]
            if len(hits) != 1:
                issue(f"{model} {var}: expected exactly 1 catalogue hit for ssp585, found {len(hits)}")
                continue
            key, cat_url = hits[0]
            member = key.split("_")[-4] if False else [p for p in key.split("_") if p.startswith("r") and "i" in p][0]
            fname = cat_url.rsplit("/", 1)[-1]
            url = OPENDAP_BASE + fname
            log(f"{model} {var}: opening {url}")
            t_pull0 = time.time()
            ds = xr.open_dataset(url, engine="pydap", decode_times=False)
            t_units = ds["time"].attrs.get("units", "")
            t_cal = ds["time"].attrs.get("calendar", "")
            calendars[var] = t_cal
            times_raw = np.asarray(ds["time"].values, dtype=float)
            dates = list(cftime.num2date(times_raw, t_units, calendar=t_cal))
            lats = np.asarray(ds["lat"].values, dtype=float); lons = np.asarray(ds["lon"].values, dtype=float)
            orig_units = ds[var].attrs.get("units", "")
            conv = conversion_for(var, orig_units)
            if conv is None:
                issue(f"{model} {var}: unrecognised units '{orig_units}' — conversion not applied, values written raw")
                factor, offset, out_units, conv_desc = 1.0, 0.0, orig_units, "NONE (unrecognised units)"
            else:
                factor, offset, out_units, conv_desc = conv
            for st, (slat, slon) in STATIONS.items():
                nc_path = out/"raw_nc"/f"{st}_{model}_{var}.nc"
                if nc_path.exists():
                    # Resume support: a series already on disk (e.g. before a
                    # runtime restart) is read back locally, never re-pulled.
                    loc = None
                    try:
                        loc = xr.open_dataset(nc_path, decode_times=False)
                        arr_c = np.asarray(loc[var].values, dtype=float)
                        t_c = list(cftime.num2date(np.asarray(loc["time"].values, dtype=float),
                                                    loc["time"].attrs.get("units", ""), calendar=loc["time"].attrs.get("calendar", "")))
                        iy = int(loc.attrs.get("grid_cell_lat_index", -1)); ix = int(loc.attrs.get("grid_cell_lon_index", -1))
                        cell_lat, cell_lon = float(loc["lat"].values), float(loc["lon"].values)
                        pull_time = str(loc.attrs.get("download_time_utc", "cached"))
                        loc.close(); loc = None
                        if iy >= 0 and len(arr_c) == len(dates) and all((a.year, a.month, a.day) == (b.year, b.month, b.day) for a, b in zip(t_c, dates)):
                            dist = haversine_km(slat, slon, cell_lat, cell_lon)
                            sha = hashlib.sha256(nc_path.read_bytes()).hexdigest()
                            feb29 = sum(1 for d in dates if d.month == 2 and d.day == 29)
                            nmiss = int(np.isnan(arr_c).sum())
                            vals_out_c = arr_c * factor + offset
                            finite = vals_out_c[~np.isnan(vals_out_c)]
                            meta_rows.append({
                                "station": st, "model": model, "member": member, "variable": var,
                                "calendar": t_cal, "units_original": orig_units, "units_output": out_units,
                                "conversion": conv_desc, "url": url, "grid_cell_lat_index": iy, "grid_cell_lon_index": ix,
                                "cell_lat": f"{cell_lat:.6f}", "cell_lon": f"{cell_lon:.6f}",
                                "distance_to_station_km": f"{dist:.2f}",
                                "first_date": f"{dates[0].year:04d}-{dates[0].month:02d}-{dates[0].day:02d}",
                                "last_date": f"{dates[-1].year:04d}-{dates[-1].month:02d}-{dates[-1].day:02d}",
                                "row_count": len(dates), "feb29_rows": feb29, "missing_values": nmiss,
                                "min_output": f"{finite.min():.4f}" if len(finite) else "",
                                "max_output": f"{finite.max():.4f}" if len(finite) else "",
                                "download_time_utc": pull_time, "raw_nc_file": nc_path.name, "sha256": sha})
                            data[st][var] = {"dates": dates, "vals": vals_out_c, "calendar": t_cal}
                            log(f"{model} {var} {st}: cached on disk, resumed without re-pull")
                            continue
                    except Exception as e:
                        log(f"{model} {var} {st}: cache read failed ({e}), re-pulling")
                    finally:
                        if loc is not None:
                            try:
                                loc.close()
                            except Exception:
                                pass
                iy = int(np.argmin(np.abs(lats - slat))); ix = int(np.argmin(np.abs(lons - slon)))
                cell_lat, cell_lon = float(lats[iy]), float(lons[ix])
                dist = haversine_km(slat, slon, cell_lat, cell_lon)
                arr = None
                for attempt in range(1, 5):
                    try:
                        arr = ds[var].isel(lat=iy, lon=ix).values
                        break
                    except Exception as e:
                        log(f"{model} {var} {st}: pull attempt {attempt} failed ({type(e).__name__}: {str(e)[:160]})")
                        if attempt == 4:
                            raise
                        time.sleep(30 * attempt)
                        try:
                            ds.close()
                        except Exception:
                            pass
                        ds = xr.open_dataset(url, engine="pydap", decode_times=False)
                arr = np.ma.filled(np.ma.asarray(arr, dtype=float), np.nan) if np.ma.isMaskedArray(arr) else np.asarray(arr, dtype=float)
                pull_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
                vals_out = arr * factor + offset
                # raw subset NetCDF, full precision, time + calendar kept
                raw = xr.Dataset(
                    {var: (("time",), arr)},
                    coords={"time": ("time", times_raw), "lat": cell_lat, "lon": cell_lon},
                    attrs={"source_url": url, "model": model, "member": member, "variable": var,
                           "station": st, "station_lat": slat, "station_lon": slon,
                           "grid_cell_lat_index": iy, "grid_cell_lon_index": ix,
                           "download_time_utc": pull_time})
                raw["time"].attrs.update({"units": t_units, "calendar": t_cal})
                raw[var].attrs.update({"units": orig_units})
                raw.to_netcdf(nc_path)
                sha = hashlib.sha256(nc_path.read_bytes()).hexdigest()
                feb29 = sum(1 for d in dates if d.month == 2 and d.day == 29)
                nmiss = int(np.isnan(arr).sum())
                finite = vals_out[~np.isnan(vals_out)]
                meta_rows.append({
                    "station": st, "model": model, "member": member, "variable": var,
                    "calendar": t_cal, "units_original": orig_units, "units_output": out_units,
                    "conversion": conv_desc, "url": url, "grid_cell_lat_index": iy, "grid_cell_lon_index": ix,
                    "cell_lat": f"{cell_lat:.6f}", "cell_lon": f"{cell_lon:.6f}",
                    "distance_to_station_km": f"{dist:.2f}",
                    "first_date": f"{dates[0].year:04d}-{dates[0].month:02d}-{dates[0].day:02d}",
                    "last_date": f"{dates[-1].year:04d}-{dates[-1].month:02d}-{dates[-1].day:02d}",
                    "row_count": len(dates), "feb29_rows": feb29, "missing_values": nmiss,
                    "min_output": f"{finite.min():.4f}" if len(finite) else "",
                    "max_output": f"{finite.max():.4f}" if len(finite) else "",
                    "download_time_utc": pull_time, "raw_nc_file": nc_path.name, "sha256": sha})
                data[st][var] = {"dates": dates, "vals": vals_out, "calendar": t_cal}
            ds.close()
            log(f"{model} {var}: done in {time.time()-t_pull0:.0f}s (calendar={t_cal}, units={orig_units}, member={member})")

        if len(set(calendars.values())) > 1:
            issue(f"{model}: variables disagree on calendar: {calendars}")
        # station text files + checks
        for st in STATIONS:
            d = data[st]
            if not all(v in d for v in VARS):
                issue(f"{model} {st}: missing variable(s), station text file not written")
                continue
            dates = d["pr"]["dates"]
            for var in VARS:
                dv = d[var]["dates"]
                if len(dv) != len(dates) or any((a.year, a.month, a.day) != (b.year, b.month, b.day) for a, b in zip(dates, dv)):
                    issue(f"{model} {st}: {var} date axis differs from pr date axis")
            txt_path = out/"txt"/f"{st}_{model}.txt"
            with open(txt_path, "w") as f:
                for i, dt in enumerate(dates):
                    row = [f"{dt.year}", f"{dt.month}", f"{dt.day}"]
                    for var in VARS:
                        v = d[var]["vals"][i]
                        row.append("-99.9" if np.isnan(v) else f"{v:.1f}")
                    f.write("\t".join(row) + "\n")
            # ---- checks ----
            cal = calendars.get("pr", "")
            expected = 55115 if cal in ("365_day", "noleap") else 55152 if cal in STANDARD_CALS else None
            n = len(dates)
            feb29 = sum(1 for x in dates if x.month == 2 and x.day == 29)
            exp_feb29 = 0 if cal in ("365_day", "noleap") else 37 if cal in STANDARD_CALS else None
            if expected is not None and n != expected:
                issue(f"{model} {st}: row count {n} != expected {expected} for calendar {cal}")
            if exp_feb29 is not None and feb29 != exp_feb29:
                issue(f"{model} {st}: Feb-29 rows {feb29} != expected {exp_feb29}")
            if (dates[-1].year, dates[-1].month, dates[-1].day) != (2100, 12, 31):
                issue(f"{model} {st}: last date is {dates[-1]}, not 2100-12-31")
            keys = [(x.year, x.month, x.day) for x in dates]
            dup = len(keys) - len(set(keys))
            # gap check within the model's own calendar via cftime ordinal diffs
            ords = [cftime.date2num(x, "days since 1950-01-01", calendar=cal) for x in dates]
            gaps = sum(1 for a, b in zip(ords, ords[1:]) if b - a != 1)
            if dup or gaps:
                issue(f"{model} {st}: duplicate dates={dup}, non-1-day steps={gaps}")
            tmin, tmax = d["tasmin"]["vals"], d["tasmax"]["vals"]
            both = ~np.isnan(tmin) & ~np.isnan(tmax)
            n_bad_order = int(np.sum(tmin[both] > tmax[both]))
            if n_bad_order:
                issue(f"{model} {st}: tmin > tmax on {n_bad_order} days")
            check_lines.append(f"## {st} — {model} (calendar: {cal})")
            check_lines.append(f"- rows: {n} (expected {expected}); Feb-29 rows: {feb29} (expected {exp_feb29}); last date: {dates[-1].year}-{dates[-1].month:02d}-{dates[-1].day:02d}; duplicates: {dup}; non-1-day steps: {gaps}; tmin>tmax days: {n_bad_order}")
            for vname, series in (("tmax", tmax), ("tmin", tmin)):
                rounded = np.round(series, 1)
                best, long_runs = max_runs_and_long_runs(rounded, dates)
                check_lines.append(f"- {vname}: longest run of identical (1-dec) values: {best[1]} days at {best[0]}")
                for sd, val, ln in long_runs:
                    check_lines.append(f"  - run >=10: from {sd}, value {val}, length {ln} days")
                    issue(f"{model} {st}: {vname} run of {ln} identical values ({val}) from {sd}")
                if model == "MIROC-ES2L" and vname == "tmin":
                    zero_from_1990 = int(np.sum((rounded == 0.0) & np.array([x.year >= 1990 for x in dates])))
                    check_lines.append(f"- MIROC-ES2L tmin specific check: days with tmin == 0.0 from 1990 onward: {zero_from_1990}")
            peaks = summer_peak_doy(dates, tmax)
            (i1, md1), (i2, md2) = peaks["1951-1980"], peaks["2071-2100"]
            check_lines.append(f"- summer peak (31-day running mean of tmax): 1951-1980 DOY {i1} ({md1[0]:02d}-{md1[1]:02d}) vs 2071-2100 DOY {i2} ({md2[0]:02d}-{md2[1]:02d}); shift {i2-i1:+d} days")
            check_lines.append("")

    cols = ["station","model","member","variable","calendar","units_original","units_output","conversion","url",
            "grid_cell_lat_index","grid_cell_lon_index","cell_lat","cell_lon","distance_to_station_km",
            "first_date","last_date","row_count","feb29_rows","missing_values","min_output","max_output",
            "download_time_utc","raw_nc_file","sha256"]
    with open(out/"metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(meta_rows)
    (out/"checks_report.md").write_text("\n".join(check_lines))
    (out/"download_log.txt").write_text("\n".join(log_lines) + "\n")
    iss = ["# ISSUES — anything unexpected, reported not fixed", "",
           "- Spec section 6 (comparison with existing thesis files) is not part of this run: Alex confirmed 2026-10-19 that Jeremy will compare.",
           "- The spec's URL template assumes grid label 'gn' for all models; PCIC's catalogue uses gr/gr1 for several models. URLs here were taken from the PCIC catalogue (member and grid label as served), per the spec's instruction not to guess members.",
           ""]
    iss += [f"- {i}" for i in issues] if issues else ["- No other issues encountered."]
    (out/"ISSUES.md").write_text("\n".join(iss) + "\n")
    log(f"Run complete. Metadata rows: {len(meta_rows)}; issues: {len(issues)}")
    (out/"download_log.txt").write_text("\n".join(log_lines) + "\n")

if __name__ == "__main__":
    main()
