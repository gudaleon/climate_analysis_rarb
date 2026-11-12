#!/usr/bin/env python3
"""Compute the annual and monthly index tables for the RARB ensrank cycle.

Uses the Wu programme's own Python Climpact/climdex reimplementation
(cdx.py for the 44 annual indices and the monthly subset, spei.py for the
monthly SPI and SPEI at the 3-month and 6-month scales), unmodified. Every
series is truncated to 1950-2024 before computation: the diagnostic windows
end in 2023, and the extra year exists so that the heat-wave aspects, which
look 91 days past a year's end, see the same forward data they would see in
an untruncated series. Jobs whose output already exists are skipped, so the
script can be rerun as the CanDCS-U6 text files arrive.
"""
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/hatch/workspace/ensrank_paper/rarb_data/provenance')
sys.path.insert(1, '/home/hatch/workspace/jeremy_wu_redteam/executor_runs/wu_claude/code')
from cdx_rarb import Series, read_daily, INDICES  # noqa: E402  # cdx.py + the documented empty-precipitation guard
from spei import spei_spi  # noqa: E402

ROOT = Path('/home/hatch/workspace/ensrank_paper')
DATA = ROOT / 'rarb_data'
ANN = ROOT / 'cycle_output' / 'annual_indices'
MON = ROOT / 'cycle_output' / 'monthly_indices'
ANN.mkdir(parents=True, exist_ok=True)
MON.mkdir(parents=True, exist_ok=True)

LAT = {'WINNIPEG': 49.92, 'BRANDON': 49.91, 'PORTAGE': 49.96, 'EMERSON': 49.03, 'MORRIS': 49.35}
TOKENS = list(LAT)


def base_token(label):
    return label[:-2] if label.endswith('NE') else label


def load_daily_csv(path):
    return pd.read_csv(path)


def monthly_with_spei(d, lat, monthly):
    g = d.groupby(['year', 'month'])
    tx = g.tx.mean()
    tn = g.tn.mean()
    pr = g.pr.apply(lambda s: np.nan if s.isna().all() else s.sum(skipna=True))
    full = pd.MultiIndex.from_product(
        [range(int(d.year.min()), int(d.year.max()) + 1), range(1, 13)], names=['year', 'month'])
    tx, tn, pr = [s.reindex(full) for s in (tx, tn, pr)]
    yrs = full.get_level_values(0).values
    mos = full.get_level_values(1).values
    sp = spei_spi(tx.values, tn.values, pr.values, yrs, mos, lat)
    m = monthly.copy()
    for k, v in sp.items():
        m[k] = pd.Series(v, index=full).reindex(m.index).values
    return m


def run(job):
    tag, kind, path, lat = job
    ann_path = ANN / f'{tag}_annual.csv'
    if ann_path.exists():
        return f'skip {tag}'
    if kind == 'model':
        d = read_daily(path)
    else:
        d = load_daily_csv(path)
    d = d[(d.year >= 1950) & (d.year <= 2024)].reset_index(drop=True)
    s = Series(d, tag)
    annual = s.annual()[INDICES]
    annual.to_csv(ann_path)
    monthly = monthly_with_spei(d, lat, s.monthly())
    monthly.to_csv(MON / f'{tag}_monthly.csv')
    return f'done {tag}'


def collect_jobs():
    jobs = []
    for token in TOKENS:
        jobs.append((f'{token}_OBS_RAW', 'obs', str(DATA / 'observed_daily' / f'{token}_OBS_RAW_daily.csv'), LAT[token]))
        jobs.append((f'{token}_OBS_AHCCD', 'obs', str(DATA / 'observed_daily' / f'{token}_OBS_AHCCD_daily.csv'), LAT[token]))
    for txt in sorted((DATA / 'candcs_u6').glob('part*/txt/*.txt')):
        label = txt.name[:-4].rsplit('_', 1)[0]
        model = txt.name[:-4].rsplit('_', 1)[1]
        jobs.append((f'{label}_{model}', 'model', str(txt), LAT[base_token(label)]))
    return jobs


if __name__ == '__main__':
    jobs = collect_jobs()
    print(f'{len(jobs)} jobs', flush=True)
    with Pool(5) as pool:
        for result in pool.imap_unordered(run, jobs):
            print(result, flush=True)
    print('ALL DONE', flush=True)
