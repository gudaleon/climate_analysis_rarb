# climate_analysis_rarb

Code for the Red–Assiniboine River Basin (RARB) cycle of the **ensrank**
ensemble-change diagnostic, supporting the paper:

> Neumann, A. *ensrank: a unified ensemble-change diagnostic that reports
> direction, rank and margin together.* Manuscript in preparation, 2026.

## What this repository is

The RARB cycle reruns, on a new basin, the full computation first developed
for a CMIP6 extremes evaluation in the Canadian Lake Erie basin:

1. acquisition of downscaled CMIP6 (CanDCS-U6) daily series and of the
   observational daily records for the basin's stations;
2. computation of the ETCCDI-style climate-extreme indices, plus SPI and
   SPEI, with an independent Python reimplementation of the Climpact 3.3 /
   climdex definitions;
3. a cell-by-cell verification of the recomputed index tables against the
   first implementation's tables;
4. the unified ensemble-change diagnostic itself — direction (median of the
   per-model changes and the pooled ensemble change, kept separate), position
   (the observed change's rank among the N + 1 values and its signed margin
   against the models' range), both bounding classifications reported per
   lineage, and a stability coordinate from perturbation branches — computed
   by the `ensrank` package, which lives in
   [climate_analysis_le](https://github.com/gudaleon/climate_analysis_le).

## Stations

Five candidate stations on the Canadian side of the basin are computed in
full: Winnipeg, Brandon, Portage la Prairie, Emerson, and Morris (Manitoba).
Which of them enter the paper's headline set is decided by a mechanical
completeness standard applied to the observed daily records, stated
numerically in the paper's methods before the diagnostic results are
examined and logged as an assumption of record. All five stations are
reported in the paper's per-station annex regardless of that verdict.

## Code only

Under the author's data and code custody rule, this repository holds **code
only**. Input data are public products (Environment and Climate Change
Canada daily climate data, the Adjusted and Homogenized Canadian Climate
Data, and the CanDCS-U6 downscaled CMIP6 product), cited at source in the
paper and not redistributed here. Data tables, figures, and the manuscript
itself are held in the author's Google Drive and are not part of this
repository.

## Status

The RARB cycle is running. Code is added here as each stage is verified,
in the order listed above; nothing in this repository should be read as a
result until the paper's verification record says so.

## Licence

The licence for this code is to be decided by the author. No licence file
is included at this stage, and nothing in this repository should be read as
granting a licence.
