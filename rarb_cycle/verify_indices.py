#!/usr/bin/env python3
"""Independent second computation path for verification of the RARB cycle.

This script does not import the programme's index code. It recomputes a set
of annual indices directly from their definitions, from the staged daily
series, applying the same annual missing-data mask convention that the
Climpact/ETCCDI definitions carry (an annual value is missing when its
variable has more than 15 missing days in the year or more than 3 missing
days in any month), and compares the recomputed values cell by cell against
the primary path's annual tables. The indices covered are the ones whose
definitions are direct aggregations of the daily series: PRCPTOT, R10mm,
R20mm, RX1day, SDII, CDD, CWD, FD, ID, SU, TR, TNn, TNx, TXx, TXn, TXm,
TNm, TMm, DTR, TXge30, TNltm20 and GDDgrow10. Percentile-based,
bootstrap-based, spell-duration-threshold and heat-wave indices are not
covered by this second path, and the cycle report says so.

Usage: verify_indices.py TAG=DAILY_CSV [TAG=DAILY_CSV ...]
where TAG names the annual table {TAG}_annual.csv in cycle_output.
Daily CSVs have columns year, month, day, pr, tx, tn; model text files in
the CanDCS-U6 form (whitespace separated, -99.9 for missing) are also
accepted when the path ends in .txt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/hatch/workspace/ensrank_paper')
ANN = ROOT / 'cycle_output' / 'annual_indices'
OUT = ROOT / 'rarb_data' / 'pipeline_checks'
TOL = 0.01

VAR_OF = {}
for ix in ('PRCPTOT', 'R10mm', 'R20mm', 'RX1day', 'SDII', 'CDD', 'CWD'):
    VAR_OF[ix] = 'pr'
for ix in ('FD', 'TNn', 'TNx', 'TNm', 'TR', 'TNltm20'):
    VAR_OF[ix] = 'tn'
for ix in ('ID', 'SU', 'TXx', 'TXn', 'TXm', 'TXge30'):
    VAR_OF[ix] = 'tx'
VAR_OF['TMm'] = 'tm'
VAR_OF['DTR'] = 'tm'
VAR_OF['GDDgrow10'] = 'tm'


def load_daily(path):
    if str(path).endswith('.txt'):
        d = pd.read_csv(path, sep=r'\s+', header=None, names=['year', 'month', 'day', 'pr', 'tx', 'tn'])
        for c in ('pr', 'tx', 'tn'):
            d.loc[d[c] <= -99.8, c] = np.nan
        return d
    return pd.read_csv(path)


def annual_mask(df, var):
    """True per year when the year passes the missing-data convention."""
    col = (df['tx'] + df['tn']) / 2 if var == 'tm' else df[var]
    miss = col.isna()
    ok = {}
    for (y), idx in df.groupby('year').groups.items():
        m = miss.loc[idx]
        if int(m.sum()) > 15:
            ok[y] = False
            continue
        months = df.loc[idx, 'month']
        ok[y] = bool((m.groupby(months).sum() <= 3).all())
    return ok


def spell_max_years(cond):
    """Maximum run length of True, credited to the year in which the run
    ends; runs may span year boundaries (the climdex spell convention)."""
    years = {}
    run = 0
    last_year = None
    for value, year in cond:
        if value:
            run += 1
        else:
            if run and last_year is not None:
                years[last_year] = max(years.get(last_year, 0), run)
            run = 0
        last_year = year
    if run and last_year is not None:
        years[last_year] = max(years.get(last_year, 0), run)
    return years


def recompute(df):
    df = df.copy()
    df['tm'] = (df['tx'] + df['tn']) / 2
    masks = {v: annual_mask(df, v) for v in ('pr', 'tx', 'tn', 'tm')}
    out = {}
    for year, g in df.groupby('year'):
        pr, tx, tn, tm = g['pr'], g['tx'], g['tn'], g['tm']
        wet = pr[pr >= 1]
        row = {}
        row['PRCPTOT'] = wet.sum()
        row['R10mm'] = int((pr >= 10).sum())
        row['R20mm'] = int((pr >= 20).sum())
        row['RX1day'] = pr.max()
        row['SDII'] = wet.sum() / len(wet) if len(wet) else 0.0
        row['FD'] = int((tn < 0).sum())
        row['ID'] = int((tx < 0).sum())
        row['SU'] = int((tx > 25).sum())
        row['TR'] = int((tn > 20).sum())
        row['TNn'] = tn.min()
        row['TNx'] = tn.max()
        row['TXx'] = tx.max()
        row['TXn'] = tx.min()
        row['TXm'] = tx.mean()
        row['TNm'] = tn.mean()
        row['TMm'] = tm.mean()
        row['DTR'] = (tx - tn).mean()
        row['TXge30'] = int((tx >= 30).sum())
        row['TNltm20'] = int((tn < -20).sum())
        row['GDDgrow10'] = float(np.where(tm > 10, tm - 10, 0).sum())
        for ix, v in VAR_OF.items():
            if ix in row and not masks[v].get(year, False):
                row[ix] = np.nan
        out[year] = row
    cdd = spell_max_years([(bool(p < 1) and not pd.isna(p), y) for p, y in zip(df['pr'], df['year'])])
    cwd = spell_max_years([(bool(p >= 1) and not pd.isna(p), y) for p, y in zip(df['pr'], df['year'])])
    for year in out:
        out[year]['CDD'] = float(cdd.get(year, 0)) if masks['pr'].get(year, False) else np.nan
        out[year]['CWD'] = float(cwd.get(year, 0)) if masks['pr'].get(year, False) else np.nan
    return pd.DataFrame(out).T


def main():
    rows = []
    total = 0
    matched = 0
    for arg in sys.argv[1:]:
        tag, path = arg.split('=', 1)
        primary = pd.read_csv(ANN / f'{tag}_annual.csv', index_col=0)
        second = recompute(load_daily(path))
        for ix in VAR_OF:
            for year in primary.index:
                if year not in second.index:
                    continue
                a = primary.loc[year, ix]
                b = second.loc[year, ix]
                total += 1
                both_nan = pd.isna(a) and pd.isna(b)
                close = (not pd.isna(a)) and (not pd.isna(b)) and abs(float(a) - float(b)) <= TOL
                if both_nan or close:
                    matched += 1
                else:
                    rows.append({'tag': tag, 'index': ix, 'year': int(year),
                                 'primary': a, 'second_path': b})
        print(f'{tag}: compared so far {total}, matched {matched}')
    pd.DataFrame(rows).to_csv(OUT / 'verification_disagreements.csv', index=False)
    (OUT / 'verification_summary.txt').write_text(
        f'cells compared: {total}\ncells matching (both missing, or within {TOL}): {matched}\n'
        f'cells disagreeing: {total - matched}\n')
    print(f'TOTAL {total} MATCHED {matched} DISAGREE {total - matched}')


if __name__ == '__main__':
    main()
