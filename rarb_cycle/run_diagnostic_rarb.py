#!/usr/bin/env python3
"""Run the ensrank unified diagnostic for the RARB cycle and write its outputs.

Frames are built from the annual index tables in cycle_output/annual_indices.
The primary record covers all five places. The headline record covers the
places that passed the completeness rule (read from
pipeline_checks/record_verdicts.csv, never hard-coded) with the Clause 3
case-level filter applied and flagged on every row. Two perturbation
branches are run: the AHCCD alternative observational layer (four places;
Morris is excluded from that comparison because its AHCCD layer carries no
precipitation series, and the exclusion is stated in the outputs) and the
one-step north-east grid-cell shift (all five places). Glyph figures are
written for the headline record and for the all-places record.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/hatch/workspace/github_sync/climate_analysis_le/ensrank/src')
from ensrank import (annual_table_to_tidy, combine_stability, diagnose,  # noqa: E402
                     plot_group_figures, run_perturbation, stability_field,
                     summary_text, tally)

ROOT = Path('/home/hatch/workspace/ensrank_paper')
ANN = ROOT / 'cycle_output' / 'annual_indices'
OUT = ROOT / 'cycle_output'
SUM = ROOT / 'rarb_data' / 'pipeline_checks'
FIG = ROOT / 'figures'
BASE, WINDOW = (1950, 1986), (1987, 2023)
MODELS = ["ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5", "CNRM-ESM2-1", "EC-Earth3-Veg",
          "GFDL-ESM4", "INM-CM5-0", "MIROC-ES2L", "MRI-ESM2-0", "NorESM2-LM"]
DISPLAY = {'WINNIPEG': 'Winnipeg', 'BRANDON': 'Brandon', 'PORTAGE': 'Portage la Prairie',
           'EMERSON': 'Emerson', 'MORRIS': 'Morris'}

proto = pd.read_csv('/home/hatch/workspace/jeremy_wu_redteam/unified_diagnostic/unified_cases.csv')
FAMILY = dict(zip(proto['index'], proto['process_family']))


def table(tag):
    return pd.read_csv(ANN / f'{tag}_annual.csv', index_col=0)


def tidy_for(obs_source, model_label):
    frames = []
    for token, display in DISPLAY.items():
        frames.append(annual_table_to_tidy(table(f'{token}_{obs_source}'), station=display,
                                           kind='observed', family_map=FAMILY))
        for model in MODELS:
            frames.append(annual_table_to_tidy(table(f'{model_label(token)}_{model}'), station=display,
                                               kind='model', model=model, family_map=FAMILY))
    return pd.concat(frames, ignore_index=True)


def headline_stations():
    v = pd.read_csv(SUM / 'record_verdicts.csv')
    rows = v[v['clause'] == 'HEADLINE SET']
    return sorted(rows.loc[rows['result'] == 'IN', 'place'].tolist())


def clause3_counts():
    rows = []
    for token, display in DISPLAY.items():
        t = table(f'{token}_OBS_RAW')
        for ix in t.columns:
            rows.append({'station': display, 'index': ix,
                         'obs_valid_years_historical': int(t.loc[1950:1986, ix].notna().sum()),
                         'obs_valid_years_present': int(t.loc[1987:2023, ix].notna().sum())})
    c = pd.DataFrame(rows)
    c['clause3_pass'] = (c['obs_valid_years_historical'] >= 30) & (c['obs_valid_years_present'] >= 30)
    return c


def main():
    primary_frame = tidy_for('OBS_RAW', lambda tok: tok)
    record = diagnose(primary_frame, base=BASE, window=WINDOW)
    record.to_csv(OUT / 'rarb_record_all_stations.csv', index=False)

    # --- grid-cell shift branch (all five places) ---
    ne_frame = tidy_for('OBS_RAW', lambda tok: tok + 'NE')
    _, _, ne_comp = run_perturbation(primary_frame, ne_frame, base=BASE, window=WINDOW)
    ne_comp.to_csv(OUT / 'branch_gridcell_shift_comparison.csv', index=False)
    ne_field = stability_field(ne_comp, 'one-step NE grid-cell shift')

    # --- AHCCD branch (four places; Morris excluded, stated in outputs) ---
    four = [d for d in DISPLAY.values() if d != 'Morris']
    base4 = primary_frame[primary_frame['station'].isin(four)]
    ahccd_frames = []
    for token, display in DISPLAY.items():
        if display not in four:
            continue
        ahccd_frames.append(annual_table_to_tidy(table(f'{token}_OBS_AHCCD'), station=display,
                                                 kind='observed', family_map=FAMILY))
        for model in MODELS:
            ahccd_frames.append(annual_table_to_tidy(table(f'{token}_{model}'), station=display,
                                                     kind='model', model=model, family_map=FAMILY))
    ahccd_frame = pd.concat(ahccd_frames, ignore_index=True)
    _, ahccd_record, ahccd_comp = run_perturbation(base4, ahccd_frame, base=BASE, window=WINDOW)
    ahccd_comp.to_csv(OUT / 'branch_ahccd_comparison.csv', index=False)
    ahccd_record.to_csv(OUT / 'branch_ahccd_record.csv', index=False)
    ahccd_field = stability_field(ahccd_comp, 'AHCCD homogenised layer')

    # --- combined stability field on the all-places record ---
    record = record.set_index(['station', 'index'])
    flips = []
    for key in record.index:
        labels = []
        if key in ne_field.index and ne_field.loc[key]:
            labels.append(ne_field.loc[key])
        if key in ahccd_field.index and ahccd_field.loc[key]:
            labels.append(ahccd_field.loc[key])
        flips.append('; '.join(labels))
    record['perturbation_flip'] = flips
    c3 = clause3_counts().set_index(['station', 'index'])
    record['obs_valid_years_historical'] = [c3.loc[k, 'obs_valid_years_historical'] for k in record.index]
    record['obs_valid_years_present'] = [c3.loc[k, 'obs_valid_years_present'] for k in record.index]
    record['clause3_pass'] = [bool(c3.loc[k, 'clause3_pass']) for k in record.index]
    hs = headline_stations()
    record['headline_station'] = [s in hs for s in record.index.get_level_values(0)]
    record = record.reset_index()
    record.to_csv(OUT / 'rarb_unified_cases.csv', index=False)

    # --- headline record and tallies ---
    headline = record[record['headline_station'] & record['clause3_pass']].copy()
    headline.to_csv(OUT / 'rarb_headline_cases.csv', index=False)
    tally_frames = []
    for scheme in ('bounding_class_613', 'bounding_class_v7'):
        t = tally(headline, scheme=scheme)
        t.insert(0, 'scope', 'headline set, both stations')
        t.insert(1, 'scheme', scheme)
        tally_frames.append(t)
        for station in hs:
            t = tally(headline[headline['station'] == station], scheme=scheme)
            t.insert(0, 'scope', f'headline station: {station}')
            t.insert(1, 'scheme', scheme)
            tally_frames.append(t)
        for family, group in headline.groupby('process_family'):
            t = tally(group, scheme=scheme)
            t.insert(0, 'scope', f'process family: {family}')
            t.insert(1, 'scheme', scheme)
            tally_frames.append(t)
    # all-places and per-station annex tallies (6.13 scheme), no Clause 3 filter
    for scheme in ('bounding_class_613', 'bounding_class_v7'):
        t = tally(record, scheme=scheme)
        t.insert(0, 'scope', 'all five stations (annex record)')
        t.insert(1, 'scheme', scheme)
        tally_frames.append(t)
        for station, group in record.groupby('station'):
            t = tally(group, scheme=scheme)
            t.insert(0, 'scope', f'station annex: {station}')
            t.insert(1, 'scheme', scheme)
            tally_frames.append(t)
    pd.concat(tally_frames, ignore_index=True).to_csv(OUT / 'rarb_tallies.csv', index=False)

    (OUT / 'rarb_summary_text.txt').write_text(
        'HEADLINE RECORD\n' + summary_text(headline) + '\n\nALL-PLACES RECORD\n' + summary_text(record) + '\n')
    written = plot_group_figures(headline, FIG)
    written += plot_group_figures(record, FIG / 'annex_all_stations')
    print('figures:', [str(w) for w in written])
    print('headline stations:', hs)
    print('headline cases:', len(headline), 'of', len(record))
    print(summary_text(headline))


if __name__ == '__main__':
    main()
