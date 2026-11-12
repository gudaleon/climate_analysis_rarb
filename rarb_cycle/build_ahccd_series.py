#!/usr/bin/env python3
"""Extract and parse the AHCCD daily series for the five RARB candidate places.

The AHCCD daily products store one text file per joined station, with one row
per month and 31 day values per row, each value carrying an optional flag
letter and with -9999.9 (temperature) or -9999.99 (precipitation) marking a
missing value. This script extracts the candidate files from the staged
archives unchanged, parses them into daily series, combines maximum and
minimum temperature with adjusted total precipitation into one daily table
per place over 1950-2023, and writes the per-variable valid-year support
counts under the same annual mask convention used for the primary record.
Morris has no adjusted precipitation file in the product, so its AHCCD
precipitation column is entirely missing, which is reported, not filled.
"""
import re
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('/home/hatch/workspace/ensrank_paper/rarb_data')
RAW = ROOT / 'ahccd' / 'raw'
EXTRACT = ROOT / 'ahccd' / 'extracted'
EXTRACT.mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'observed_daily'
SUM = ROOT / 'pipeline_checks'

TEMP_ARCHIVE = 'Homog_daily_{kind}_temp_v2023_Gen3.zip'
PRECIP_ARCHIVE = 'Adj_Daily_Total_v2017.zip'
PLACES = {
    'WINNIPEG': ('Winnipeg', '502S001', '5023222'),
    'BRANDON': ('Brandon', '5010490', '5010480'),
    'PORTAGE': ('Portage la Prairie', '5012322', '5012321'),
    'EMERSON': ('Emerson', '5020881', '5020882'),
    'MORRIS': ('Morris', '5021920', None),
}
TOKEN_RE = re.compile(r'(-?\d+\.\d+)[A-Za-z]?')


def extract_member(archive, member):
    target = EXTRACT / member
    with zipfile.ZipFile(RAW / archive) as z:
        data = z.read(member)
    target.write_bytes(data)
    return data.decode('utf-8', errors='replace')


def parse_monthly_rows(text):
    """Return a daily Series from an AHCCD monthly-row text file."""
    records = {}
    for line in text.splitlines():
        if len(line) < 8 or not line.strip()[:4].isdigit():
            continue
        try:
            year = int(line[0:5])
            month = int(line[5:8])
        except ValueError:
            continue
        if not (1800 <= year <= 2100 and 1 <= month <= 12):
            continue
        tokens = TOKEN_RE.findall(line[8:])
        if len(tokens) != 31:
            continue
        for day, tok in enumerate(tokens, start=1):
            try:
                date = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue
            value = float(tok)
            records[date] = np.nan if value <= -9999.0 else value
    s = pd.Series(records, dtype=float).sort_index()
    return s


def valid_year_count(s, y0, y1):
    full = pd.date_range(f'{y0}-01-01', f'{y1}-12-31', freq='D')
    miss = s.reindex(full).isna()
    years, months = full.year, full.month
    n = 0
    for y in range(y0, y1 + 1):
        sel = years == y
        if int(miss[sel].sum()) > 15:
            continue
        if bool((pd.Series(miss[sel]).groupby(months[sel]).sum() <= 3).all()):
            n += 1
    return n


rows = []
for token, (display, temp_id, precip_id) in PLACES.items():
    tx = parse_monthly_rows(extract_member(TEMP_ARCHIVE.format(kind='max'), f'dx{temp_id}.txt'))
    tn = parse_monthly_rows(extract_member(TEMP_ARCHIVE.format(kind='min'), f'dn{temp_id}.txt'))
    if precip_id is not None:
        pr = parse_monthly_rows(extract_member(PRECIP_ARCHIVE, f'dt{precip_id}.txt'))
    else:
        pr = pd.Series(dtype=float)
    full = pd.date_range('1950-01-01', '2023-12-31', freq='D')
    df = pd.DataFrame({'year': full.year, 'month': full.month, 'day': full.day,
                       'pr': pr.reindex(full).values, 'tx': tx.reindex(full).values,
                       'tn': tn.reindex(full).values})
    df.to_csv(OUT / f'{token}_OBS_AHCCD_daily.csv', index=False)
    for var, s in (('pr', pr), ('tx', tx), ('tn', tn)):
        ss = s.reindex(full)
        first = str(ss.first_valid_index().date()) if ss.first_valid_index() is not None else ''
        last = str(ss.last_valid_index().date()) if ss.last_valid_index() is not None else ''
        rows.append({'place': display, 'variable': var,
                     'ahccd_id': precip_id if var == 'pr' else temp_id,
                     'first_value': first, 'last_value': last,
                     'valid_years_historical_1950_1986': valid_year_count(ss, 1950, 1986),
                     'valid_years_present_1987_2023': valid_year_count(ss, 1987, 2023)})
    print(token, 'done')
pd.DataFrame(rows).to_csv(SUM / 'ahccd_support.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
