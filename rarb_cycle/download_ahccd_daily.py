#!/usr/bin/env python3
"""Download the official AHCCD daily archives listed in the CRD directory.
Data acquisition only: archives are staged unchanged."""
import hashlib, urllib.request
from pathlib import Path
BASE='https://crd-data-donnees-rdc.ec.gc.ca/CDAS/products/AHCCD_daily/'
FILES=['Homog_daily_max_temp_v2023_Gen3.zip','Homog_daily_min_temp_v2023_Gen3.zip','Homog_daily_mean_temp_v2023_Gen3.zip','Adj_Daily_Total_v2017.zip','Adj_Daily_Rain_v2017.zip','Adj_Daily_Snow_v2017.zip','Homog_Temperature_Stations_Segmented_List_Gen3.xls','Adj_Precipitation_Stations.xls','Homog_Temperature_Documentation_Daily_Gen3.doc','Adj_Precipitation_Documentation_Daily.doc']
OUT=Path('/home/hatch/workspace/ensrank_paper/rarb_data/ahccd/raw'); OUT.mkdir(parents=True,exist_ok=True)
for fn in FILES:
    p=OUT/fn
    if p.exists() and p.stat().st_size>1000:
        print('cached',fn,p.stat().st_size); continue
    req=urllib.request.Request(BASE+fn,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=300) as r: data=r.read()
    p.write_bytes(data)
    print('downloaded',fn,len(data),hashlib.sha256(data).hexdigest())
