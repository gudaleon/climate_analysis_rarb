#!/usr/bin/env python3
"""Merge the five CanDCS-U6 part downloads and verify the merged staging.

Copies every part's raw NetCDF subsets and text series into the common
candcs_u6/raw_nc and candcs_u6/txt folders, concatenates the five metadata
tables, and then reads everything back: each text series is re-read and its
row count and first and last dates are compared with its metadata rows, and
each text series' values are compared against the values recomputed from
its three raw NetCDF subsets under the recorded unit conversions. It also
compares the block route's cells against the first route's per-cell
downloads for every station-model-variable the first route completed, cell
by cell. The results are written to pipeline_checks/candcs_readback.csv and
pipeline_checks/candcs_route_comparison.txt. Nothing is fixed here: a file
that fails a check is reported, not repaired.
"""
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path('/home/hatch/workspace/ensrank_paper/rarb_data')
C = ROOT / 'candcs_u6'
SUM = ROOT / 'pipeline_checks'
PARTS = [C / f'part{i}' for i in range(1, 6)]

(C / 'txt').mkdir(exist_ok=True)
(C / 'raw_nc').mkdir(exist_ok=True)
metas = []
for part in PARTS:
    for f in (part / 'txt').glob('*.txt'):
        shutil.copy2(f, C / 'txt' / f.name)
    for f in (part / 'raw_nc').glob('*.nc'):
        shutil.copy2(f, C / 'raw_nc' / f.name)
    m = pd.read_csv(part / 'metadata.csv')
    metas.append(m)
meta = pd.concat(metas, ignore_index=True)
meta.to_csv(C / 'metadata.csv', index=False)
print('metadata rows:', len(meta), 'txt files:', len(list((C / 'txt').glob('*.txt'))),
      'nc files:', len(list((C / 'raw_nc').glob('*.nc'))))

rows = []
for (station, model), group in meta.groupby(['station', 'model']):
    txt = C / 'txt' / f'{station}_{model}.txt'
    d = pd.read_csv(txt, sep=r'\s+', header=None, names=['year', 'month', 'day', 'pr', 'tx', 'tn'])
    ok_rows = len(d) == int(group['row_count'].iloc[0])
    first = f"{d.year.iloc[0]:04d}-{d.month.iloc[0]:02d}-{d.day.iloc[0]:02d}"
    last = f"{d.year.iloc[-1]:04d}-{d.month.iloc[-1]:02d}-{d.day.iloc[-1]:02d}"
    ok_dates = (first == group['first_date'].iloc[0]) and (last == group['last_date'].iloc[0])
    # value check against the raw NetCDF subsets under the recorded conversions
    ok_vals = True
    detail = ''
    for var, col in (('pr', 'pr'), ('tasmax', 'tx'), ('tasmin', 'tn')):
        grow = group[group['variable'] == var].iloc[0]
        ds = xr.open_dataset(C / 'raw_nc' / grow['raw_nc_file'], decode_times=False)
        arr = np.asarray(ds[var].values, dtype=float)
        ds.close()
        conv = grow['conversion']
        if 'x86400' in conv:
            arr = arr * 86400.0
        elif '-273.15' in conv:
            arr = arr - 273.15
        txt_vals = d[col].values.astype(float)
        both_nan = np.isnan(arr) & np.isnan(txt_vals)
        close = np.isclose(np.where(both_nan, 0, arr), np.where(both_nan, 0, txt_vals), atol=0.051)
        if not bool((both_nan | close).all()):
            ok_vals = False
            detail = f'{var}: {int((~(both_nan | close)).sum())} cells differ'
    rows.append({'station': station, 'model': model, 'rows_ok': ok_rows, 'dates_ok': ok_dates,
                 'values_ok': ok_vals, 'detail': detail})
readback = pd.DataFrame(rows)
readback.to_csv(SUM / 'candcs_readback.csv', index=False)
print(readback.groupby(['rows_ok', 'dates_ok', 'values_ok']).size())

# Route comparison: block route vs the first route's per-cell downloads.
lines = []
first_root = C / 'first_route_parts'
compared = 0
mismatched = 0
for nc in sorted(first_root.glob('part*/raw_nc/*.nc')):
    twin = C / 'raw_nc' / nc.name
    if not twin.exists():
        lines.append(f'{nc.name}: no block-route twin staged')
        continue
    a = xr.open_dataset(nc, decode_times=False)
    b = xr.open_dataset(twin, decode_times=False)
    var = [v for v in a.data_vars][0]
    av = np.asarray(a[var].values, dtype=float)
    bv = np.asarray(b[var].values, dtype=float)
    same = (av.shape == bv.shape) and bool(np.allclose(av, bv, equal_nan=True))
    compared += 1
    mismatched += 0 if same else 1
    lines.append(f'{nc.name}: shape {av.shape} vs {bv.shape}; identical values: {same}')
    a.close()
    b.close()
lines.append(f'route comparison files compared: {compared}; files with any differing value: {mismatched}')
(SUM / 'candcs_route_comparison.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
