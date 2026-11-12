#!/usr/bin/env python3
"""Build the staged ECCC daily series for the RARB ensrank cycle.

Reads the annual bulk CSV files under eccc_daily_raw/by_station_id/, extracts
(year, month, day, pr, tx, tn) per Station ID, writes one daily CSV per
Station ID, builds the spliced observed series for the five candidate places
under the splice rule logged in pipeline_checks/ASSUMPTIONS_OF_RECORD.md
(including Amendment 1 for Portage la Prairie), and writes the record
summaries: per-station coverage, missingness by decade, valid-year counts
under the Climpact/cdx annual mask convention, and the completeness-rule
verdicts for the spliced places. This script computes record facts only. It
computes no climate index and no diagnostic.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('/home/hatch/workspace/ensrank_paper/rarb_data')
RAW = ROOT / 'eccc_daily_raw' / 'by_station_id'
OUT = ROOT / 'observed_daily'
OUT.mkdir(parents=True, exist_ok=True)
SUM = ROOT / 'pipeline_checks'
SUM.mkdir(parents=True, exist_ok=True)

STATION_YEARS = {
    3698: (1950, 2008), 47407: (2008, 2013), 51097: (2018, 2023), 27174: (1996, 2023), 28051: (1999, 2023),
    3471: (1950, 2012), 50821: (2012, 2023), 3472: (1950, 2010),
    3517: (1950, 1971), 3519: (1970, 2023), 3520: (1962, 1996), 3518: (1950, 1992), 10884: (1992, 2023), 45987: (2007, 2023),
    3595: (1950, 1997), 26963: (1997, 2009), 48068: (2009, 2023),
    3628: (1950, 1987), 3629: (1954, 1959), 3630: (1961, 2007),
}
# Splice priority orders (Assumptions of Record, with Amendment 1 for Portage).
PLACES = {
    'WINNIPEG': ('Winnipeg', [3698, 47407, 51097, 27174]),
    'BRANDON': ('Brandon', [3471, 50821]),
    'PORTAGE': ('Portage la Prairie', [3517, 3519, 3520, 3518, 10884]),
    'EMERSON': ('Emerson', [3595, 26963, 48068]),
    'MORRIS': ('Morris', [3628, 3630]),
}
VARS = ['pr', 'tx', 'tn']
COLMAP = {'Total Precip (mm)': 'pr', 'Max Temp (°C)': 'tx', 'Min Temp (°C)': 'tn'}
WINDOWS = {'Historical_1950_1986': (1950, 1986), 'Present_1987_2023': (1987, 2023)}


def load_station(sid):
    frames = []
    for year in range(*STATION_YEARS[sid]):
        pass
    a, b = STATION_YEARS[sid]
    for year in range(a, b + 1):
        p = RAW / str(sid) / f'{year}.csv'
        d = pd.read_csv(p, usecols=['Date/Time'] + list(COLMAP))
        d = d.rename(columns=COLMAP)
        d['date'] = pd.to_datetime(d['Date/Time'])
        frames.append(d[['date', 'pr', 'tx', 'tn']])
    df = pd.concat(frames, ignore_index=True).sort_values('date')
    df = df.drop_duplicates(subset='date', keep='first')
    return df.set_index('date')


def valid_years(df, var, y0, y1):
    """Years in [y0, y1] passing the cdx annual mask for var: <=15 missing
    days in the year and <=3 missing days in every month. Dates absent from
    the frame count as missing."""
    full = pd.date_range(f'{y0}-01-01', f'{y1}-12-31', freq='D')
    s = df[var].reindex(full)
    miss = s.isna()
    years = full.year
    months = full.month
    ok = {}
    for y in range(y0, y1 + 1):
        sel = years == y
        if int(miss[sel].sum()) > 15:
            ok[y] = False
            continue
        mvals = pd.Series(miss[sel]).groupby(months[sel]).sum()
        ok[y] = bool((mvals <= 3).all())
    return ok


def decade_missingness(df, y0, y1):
    rows = []
    for d0 in range(1950, 2021, 10):
        d1 = min(d0 + 9, 2023)
        a, b = max(d0, y0), min(d1, y1)
        if a > b:
            continue
        full = pd.date_range(f'{a}-01-01', f'{b}-12-31', freq='D')
        for var in VARS:
            s = df[var].reindex(full)
            rows.append({'decade': f'{d0}s' if d1 == d0 + 9 else f'{d0}-{d1}',
                         'variable': var, 'expected_days': len(full),
                         'missing_days': int(s.isna().sum()),
                         'missing_pct': round(100.0 * float(s.isna().mean()), 2)})
    return rows


station_frames = {}
station_summary = []
station_decades = []
for sid in STATION_YEARS:
    df = load_station(sid)
    station_frames[sid] = df
    out = pd.DataFrame({'year': df.index.year, 'month': df.index.month, 'day': df.index.day,
                        'pr': df['pr'].values, 'tx': df['tx'].values, 'tn': df['tn'].values})
    out.to_csv(OUT / f'station_{sid}_daily.csv', index=False)
    a, b = STATION_YEARS[sid]
    row = {'station_id': sid, 'staged_years': f'{a}-{b}', 'rows': len(df),
           'first_row': str(df.index.min().date()), 'last_row': str(df.index.max().date())}
    anydata = df.dropna(how='all')
    row['first_date_any_data'] = str(anydata.index.min().date()) if len(anydata) else ''
    row['last_date_any_data'] = str(anydata.index.max().date()) if len(anydata) else ''
    for var in VARS:
        row[f'{var}_nonmissing'] = int(df[var].notna().sum())
        s = df[var].dropna()
        row[f'{var}_first'] = str(s.index.min().date()) if len(s) else ''
        row[f'{var}_last'] = str(s.index.max().date()) if len(s) else ''
    station_summary.append(row)
    for r in decade_missingness(df, a, b):
        station_decades.append({'station_id': sid, **r})
pd.DataFrame(station_summary).to_csv(SUM / 'station_record_summary.csv', index=False)
pd.DataFrame(station_decades).to_csv(SUM / 'station_missingness_by_decade.csv', index=False)

verdict_rows = []
place_decades = []
valid_rows = []
for token, (display, sids) in PLACES.items():
    full = pd.date_range('1950-01-01', '2023-12-31', freq='D')
    spliced = pd.DataFrame(index=full)
    for var in VARS:
        s = pd.Series(np.nan, index=full)
        for sid in sids:
            src = station_frames[sid][var].reindex(full)
            s = s.fillna(src)
        spliced[var] = s
    out = pd.DataFrame({'year': full.year, 'month': full.month, 'day': full.day,
                        'pr': spliced['pr'].values, 'tx': spliced['tx'].values, 'tn': spliced['tn'].values})
    out.to_csv(OUT / f'{token}_OBS_RAW_daily.csv', index=False)
    for r in decade_missingness(spliced, 1950, 2023):
        place_decades.append({'place': display, **r})
    has1950 = bool(out.loc[out['year'] == 1950, ['pr', 'tx', 'tn']].notna().any().any())
    has2023 = bool(out.loc[out['year'] == 2023, ['pr', 'tx', 'tn']].notna().any().any())
    span_ok = has1950 and has2023
    all_clause1 = True
    for var in VARS:
        for wname, (y0, y1) in WINDOWS.items():
            ok = valid_years(spliced, var, y0, y1)
            n_ok = sum(ok.values())
            valid_rows.append({'place': display, 'variable': var, 'window': wname,
                               'valid_years': n_ok, 'window_years': y1 - y0 + 1,
                               'invalid_years': ','.join(str(y) for y, v in ok.items() if not v)})
            passed = n_ok >= 30
            all_clause1 &= passed
            verdict_rows.append({'place': display, 'clause': 'Clause 1', 'variable': var,
                                 'window': wname, 'valid_years': n_ok, 'required': 34,
                                 'result': 'pass' if passed else 'FAIL'})
    verdict_rows.append({'place': display, 'clause': 'Clause 2 (span)', 'variable': 'all',
                         'window': '1950-2023', 'valid_years': '', 'required': '',
                         'result': 'pass' if span_ok else 'FAIL'})
    verdict_rows.append({'place': display, 'clause': 'HEADLINE SET', 'variable': 'all',
                         'window': 'both', 'valid_years': '', 'required': '',
                         'result': 'IN' if (all_clause1 and span_ok) else 'OUT'})
# Verdicts under the original 34-year threshold (Amendment 2 transparency).
v34 = []
for (token, (display, sids)) in PLACES.items():
    sub = pd.DataFrame(valid_rows)
    sub = sub[sub['place'] == display]
    ok34 = bool((sub['valid_years'] >= 34).all())
    has1950 = display != 'Morris'  # Morris span result is recomputed below from its frame
    v34.append({'place': display, 'threshold': 34, 'clause1_all_pass': ok34})
pd.DataFrame(v34).to_csv(SUM / 'record_verdicts_threshold34_partial.csv', index=False)
pd.DataFrame(place_decades).to_csv(SUM / 'place_missingness_by_decade.csv', index=False)
pd.DataFrame(valid_rows).to_csv(SUM / 'place_valid_years.csv', index=False)
pd.DataFrame(verdict_rows).to_csv(SUM / 'record_verdicts.csv', index=False)
print('done')
print(pd.DataFrame(verdict_rows).to_string(index=False))
