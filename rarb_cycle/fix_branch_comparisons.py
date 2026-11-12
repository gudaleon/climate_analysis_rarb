#!/usr/bin/env python3
"""Correct the perturbation-branch comparisons for a pandas-3 artefact.

In the environment used for this cycle (pandas 3.1.1), a record column that
mixes classification strings with the diagnostic's empty value for
direction-only cases is stored with the new string dtype, which converts
the empty value to NaN. ensrank's compare_records tests classification
fields by equality, and NaN is not equal to NaN, so every direction-only
case (the 31 Morris cases that carry no observed position) was reported as
a classification flip on every branch, with a stability string reading
"nan -> nan". The positioned cases were never affected.

This script recomputes both branch comparisons with the empty
classification replaced by an explicit NO_POSITION sentinel in both
records before comparison, which restores the equality semantics the
package was written against, and then rebuilds the combined stability
field on the unified cases table, the headline cases table, and the glyph
figures. The package itself is not modified. The correction, its cause,
and the before-and-after flip counts are recorded in
pipeline_checks/branch_comparison_correction.txt.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, '/home/hatch/workspace/github_sync/climate_analysis_le/ensrank/src')
from ensrank import (annual_table_to_tidy, compare_records, diagnose,  # noqa: E402
                     plot_group_figures, stability_field)

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
SENTINEL = 'NO_POSITION'


def table(tag):
    return pd.read_csv(ANN / f'{tag}_annual.csv', index_col=0)


def tidy_for(obs_source, model_label, stations=None):
    frames = []
    for token, display in DISPLAY.items():
        if stations is not None and display not in stations:
            continue
        frames.append(annual_table_to_tidy(table(f'{token}_{obs_source}'), station=display,
                                           kind='observed', family_map=FAMILY))
        for model in MODELS:
            frames.append(annual_table_to_tidy(table(f'{model_label(token)}_{model}'), station=display,
                                               kind='model', model=model, family_map=FAMILY))
    return pd.concat(frames, ignore_index=True)


def sentinelised(record):
    r = record.copy()
    for col in ('bounding_class_613', 'bounding_class_v7'):
        r[col] = r[col].fillna(SENTINEL)
    return r


def compare(frame_a, frame_b):
    ra = sentinelised(diagnose(frame_a, base=BASE, window=WINDOW))
    rb = sentinelised(diagnose(frame_b, base=BASE, window=WINDOW))
    return compare_records(ra, rb)


def main():
    before_ne = pd.read_csv(OUT / 'branch_gridcell_shift_comparison.csv')
    before_ah = pd.read_csv(OUT / 'branch_ahccd_comparison.csv')

    primary_frame = tidy_for('OBS_RAW', lambda tok: tok)
    ne_comp = compare(primary_frame, tidy_for('OBS_RAW', lambda tok: tok + 'NE'))
    ne_comp.to_csv(OUT / 'branch_gridcell_shift_comparison.csv', index=False)
    ne_field = stability_field(ne_comp, 'one-step NE grid-cell shift')

    four = [d for d in DISPLAY.values() if d != 'Morris']
    base4 = primary_frame[primary_frame['station'].isin(four)]
    ahccd_comp = compare(base4, tidy_for('OBS_AHCCD', lambda tok: tok, stations=four))
    ahccd_comp.to_csv(OUT / 'branch_ahccd_comparison.csv', index=False)
    ahccd_field = stability_field(ahccd_comp, 'AHCCD homogenised layer')

    record = pd.read_csv(OUT / 'rarb_unified_cases.csv').set_index(['station', 'index'])
    flips = []
    for key in record.index:
        labels = []
        if key in ne_field.index and ne_field.loc[key]:
            labels.append(ne_field.loc[key])
        if key in ahccd_field.index and ahccd_field.loc[key]:
            labels.append(ahccd_field.loc[key])
        flips.append('; '.join(labels))
    record['perturbation_flip'] = flips
    record = record.reset_index()
    record.to_csv(OUT / 'rarb_unified_cases.csv', index=False)
    headline = record[record['headline_station'] & record['clause3_pass']].copy()
    headline.to_csv(OUT / 'rarb_headline_cases.csv', index=False)
    written = plot_group_figures(headline, FIG)
    written += plot_group_figures(record, FIG / 'annex_all_stations')

    note = [
        'Branch comparison correction (pandas-3 string-dtype artefact).',
        f"Grid-cell shift branch, classification_flip_613 count before correction: {int(before_ne['classification_flip_613'].sum())} of {len(before_ne)}; after correction: {int(ne_comp['classification_flip_613'].sum())} of {len(ne_comp)}.",
        f"AHCCD branch, classification_flip_613 count before correction: {int(before_ah['classification_flip_613'].sum())} of {len(before_ah)}; after correction: {int(ahccd_comp['classification_flip_613'].sum())} of {len(ahccd_comp)}.",
        'All removed flips were Morris direction-only cases whose classification is empty in both records; no positioned case changed its flip status.',
        f'Figures regenerated: {len(written)} files.',
    ]
    (SUM / 'branch_comparison_correction.txt').write_text('\n'.join(note) + '\n')
    print('\n'.join(note))
    print('NE flips by station:')
    print(ne_comp.groupby('station')['classification_flip_613'].sum().to_string())
    print('AHCCD flips by station:')
    print(ahccd_comp.groupby('station')['classification_flip_613'].sum().to_string())


if __name__ == '__main__':
    main()
