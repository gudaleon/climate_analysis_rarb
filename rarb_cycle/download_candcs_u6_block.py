#!/usr/bin/env python3
"""CanDCS-U6 station extraction for the RARB cycle, block-pull route.

This script produces exactly the outputs of the Wu programme's per-cell
extractor (download_candcs_u6_rarb.py, itself the programme's
download_pcic.py with the RARB station table): one raw NetCDF subset and
one row of metadata per station, model and variable, one text series per
station and model, and the same section-5 checks. The difference is the
transfer pattern only. The per-cell route issues one OPeNDAP request per
station cell, and on 2026-11-12 those requests took roughly four to eight
minutes each against this server, which puts the 300 extracts of this
cycle beyond a working day. This route instead issues one OPeNDAP request
per model and variable for the single latitude-longitude block that
contains all ten target cells (latitude indices 90 to 110 and longitude
indices 485 to 535 of the product grid), and then reads the ten cells out
of the returned block locally. A timed test of this route on the
CNRM-ESM2-1 precipitation file returned the full block in 229.4 seconds.
No value is changed by the route: the cells extracted from the block are
the product's own values at the same grid indices the per-cell route
selects, the unit conversions are the programme's own (imported from the
programme's script), the dates are decoded from each file's own time
variable and calendar, and the checks are the programme's own functions.
"""
import argparse, csv, hashlib, json, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cftime
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from download_candcs_u6_rarb import (CATALOG_URL, MODELS, OPENDAP_BASE, STATIONS,  # noqa: E402
                                     VARS, conversion_for, haversine_km,
                                     max_runs_and_long_runs, summer_peak_doy)

LAT_SLICE = slice(90, 111)
LON_SLICE = slice(485, 536)
STANDARD_CALS = {"standard", "gregorian", "proleptic_gregorian"}

log_lines, issues = [], []


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


def issue(msg):
    log(f"ISSUE: {msg}")
    issues.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out = Path(args.out)
    (out / "raw_nc").mkdir(parents=True, exist_ok=True)
    (out / "txt").mkdir(parents=True, exist_ok=True)
    log(f"Block-route run start. Models: {models}")

    with urllib.request.urlopen(CATALOG_URL, timeout=120) as r:
        catalog = json.load(r)
    log(f"Catalogue entries: {len(catalog)}")

    meta_rows, check_lines = [], ["# Section-5 checks (reported, not fixed)\n"]
    for model in models:
        data = {s: {} for s in STATIONS}
        calendars = {}
        for var in VARS:
            hits = [(k, v) for k, v in catalog.items()
                    if model in k and var == k.split("_")[0] and "ssp585" in k]
            if len(hits) != 1:
                issue(f"{model} {var}: expected exactly 1 catalogue hit for ssp585, found {len(hits)}")
                continue
            key, cat_url = hits[0]
            member = [p for p in key.split("_") if p.startswith("r") and "i" in p][0]
            fname = cat_url.rsplit("/", 1)[-1]
            url = OPENDAP_BASE + fname
            log(f"{model} {var}: opening {url}")
            t0 = time.time()
            ds = xr.open_dataset(url, engine="pydap", decode_times=False)
            t_units = ds["time"].attrs.get("units", "")
            t_cal = ds["time"].attrs.get("calendar", "")
            calendars[var] = t_cal
            times_raw = np.asarray(ds["time"].values, dtype=float)
            dates = list(cftime.num2date(times_raw, t_units, calendar=t_cal))
            lats = np.asarray(ds["lat"].values, dtype=float)
            lons = np.asarray(ds["lon"].values, dtype=float)
            orig_units = ds[var].attrs.get("units", "")
            conv = conversion_for(var, orig_units)
            if conv is None:
                issue(f"{model} {var}: unrecognised units '{orig_units}' — conversion not applied, values written raw")
                factor, offset, out_units, conv_desc = 1.0, 0.0, orig_units, "NONE (unrecognised units)"
            else:
                factor, offset, out_units, conv_desc = conv
            targets = {}
            for st, (slat, slon) in STATIONS.items():
                iy = int(np.argmin(np.abs(lats - slat)))
                ix = int(np.argmin(np.abs(lons - slon)))
                targets[st] = (iy, ix)
            need_lat = sorted({iy for iy, _ in targets.values()})
            need_lon = sorted({ix for _, ix in targets.values()})
            if not (LAT_SLICE.start <= min(need_lat) and max(need_lat) < LAT_SLICE.stop
                    and LON_SLICE.start <= min(need_lon) and max(need_lon) < LON_SLICE.stop):
                issue(f"{model} {var}: a target cell falls outside the staged block slices; file skipped")
                ds.close()
                continue
            block = None
            for attempt in range(1, 5):
                try:
                    block = ds[var].isel(lat=LAT_SLICE, lon=LON_SLICE).values
                    break
                except Exception as e:
                    log(f"{model} {var}: block pull attempt {attempt} failed ({type(e).__name__}: {str(e)[:160]})")
                    if attempt == 4:
                        raise
                    time.sleep(30 * attempt)
                    try:
                        ds.close()
                    except Exception:
                        pass
                    ds = xr.open_dataset(url, engine="pydap", decode_times=False)
            block = np.ma.filled(np.ma.asarray(block, dtype=float), np.nan) if np.ma.isMaskedArray(block) else np.asarray(block, dtype=float)
            log(f"{model} {var}: block pulled in {time.time()-t0:.0f}s, shape {block.shape}")
            for st, (iy, ix) in targets.items():
                arr = block[:, iy - LAT_SLICE.start, ix - LON_SLICE.start]
                cell_lat, cell_lon = float(lats[iy]), float(lons[ix])
                slat, slon = STATIONS[st]
                dist = haversine_km(slat, slon, cell_lat, cell_lon)
                pull_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
                vals_out = arr * factor + offset
                nc_path = out / "raw_nc" / f"{st}_{model}_{var}.nc"
                raw = xr.Dataset(
                    {var: (("time",), arr)},
                    coords={"time": ("time", times_raw), "lat": cell_lat, "lon": cell_lon},
                    attrs={"source_url": url, "model": model, "member": member, "variable": var,
                           "station": st, "station_lat": slat, "station_lon": slon,
                           "grid_cell_lat_index": iy, "grid_cell_lon_index": ix,
                           "download_time_utc": pull_time,
                           "extraction_route": "single block pull per model-variable (see script docstring)"})
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
        if len(set(calendars.values())) > 1:
            issue(f"{model}: variables disagree on calendar: {calendars}")
        for st in STATIONS:
            d = data[st]
            if not all(v in d for v in VARS):
                issue(f"{model} {st}: missing variable(s), station text file not written")
                continue
            dates = d["pr"]["dates"]
            txt_path = out / "txt" / f"{st}_{model}.txt"
            with open(txt_path, "w") as f:
                for i, dt in enumerate(dates):
                    row = [f"{dt.year}", f"{dt.month}", f"{dt.day}"]
                    for var in VARS:
                        v = d[var]["vals"][i]
                        row.append("-99.9" if np.isnan(v) else f"{v:.1f}")
                    f.write("\t".join(row) + "\n")
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
            peaks = summer_peak_doy(dates, tmax)
            (i1, md1), (i2, md2) = peaks["1951-1980"], peaks["2071-2100"]
            check_lines.append(f"- summer peak (31-day running mean of tmax): 1951-1980 DOY {i1} ({md1[0]:02d}-{md1[1]:02d}) vs 2071-2100 DOY {i2} ({md2[0]:02d}-{md2[1]:02d}); shift {i2-i1:+d} days")
            check_lines.append("")
    cols = ["station", "model", "member", "variable", "calendar", "units_original", "units_output", "conversion", "url",
            "grid_cell_lat_index", "grid_cell_lon_index", "cell_lat", "cell_lon", "distance_to_station_km",
            "first_date", "last_date", "row_count", "feb29_rows", "missing_values", "min_output", "max_output",
            "download_time_utc", "raw_nc_file", "sha256"]
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(meta_rows)
    (out / "checks_report.md").write_text("\n".join(check_lines))
    iss = ["# ISSUES — anything unexpected, reported not fixed", "",
           "- Extraction route: the block-pull route documented in download_candcs_u6_block.py replaces the per-cell pull pattern of the programme's original script for transfer speed only; values, conversions, dates and checks are unchanged.",
           ""]
    iss += [f"- {i}" for i in issues] if issues else ["- No other issues encountered."]
    (out / "ISSUES.md").write_text("\n".join(iss) + "\n")
    log(f"Run complete. Metadata rows: {len(meta_rows)}; issues: {len(issues)}")
    (out / "download_log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
