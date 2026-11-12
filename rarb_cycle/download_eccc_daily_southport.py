#!/usr/bin/env python3
"""Additional ECCC daily downloads added by Amendment 1 to the assumptions of
record: the Portage la Prairie area temperature stations (Southport) and the
Portage Romance station, staged for inventory completeness."""
import concurrent.futures, csv, time, urllib.parse, urllib.request
from pathlib import Path
BASE = Path('/home/hatch/workspace/ensrank_paper/rarb_data/eccc_daily_raw/by_station_id')
LOG = Path('/home/hatch/workspace/ensrank_paper/rarb_data/provenance/eccc_daily_download_log_southport.csv')
JOBS = {3518:(1950,1992), 10884:(1992,2023), 45987:(2007,2023)}
def fetch(job):
    sid, year = job
    out = BASE/str(sid)/f'{year}.csv'
    if out.exists() and out.stat().st_size > 100:
        return (sid,year,'cached',out.stat().st_size,'')
    params = {'format':'csv','stationID':sid,'Year':year,'Month':1,'Day':1,'timeframe':2,'submit':'Download Data'}
    url = 'https://climate.weather.gc.ca/climate_data/bulk_data_e.html?' + urllib.parse.urlencode(params)
    last=''
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req,timeout=90) as r: data=r.read()
            text=data.decode('utf-8-sig',errors='strict')
            if 'Date/Time' not in text[:2000]: raise ValueError('response lacks Date/Time header')
            out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(data)
            return (sid,year,'downloaded',len(data),url)
        except Exception as e:
            last=f'{type(e).__name__}: {e}'; time.sleep(2*(attempt+1))
    return (sid,year,'FAILED',0,last)
jobs=[(sid,y) for sid,(a,b) in JOBS.items() for y in range(a,b+1)]
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    for r in ex.map(fetch,jobs): rows.append(r)
with LOG.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['station_id','year','status','bytes','url_or_error']); w.writerows(sorted(rows))
print('jobs',len(rows),'failed',sum(r[2]=='FAILED' for r in rows))
