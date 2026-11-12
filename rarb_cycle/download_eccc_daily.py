#!/usr/bin/env python3
"""Download ECCC daily bulk CSV files, one file per station ID and year.
This is inventory acquisition only. It does not alter values or combine station IDs.
"""
import concurrent.futures, csv, io, sys, time, urllib.parse, urllib.request
from pathlib import Path
BASE = Path('/home/hatch/workspace/ensrank_paper/rarb_data/eccc_daily_raw/by_station_id')
LOG = Path('/home/hatch/workspace/ensrank_paper/rarb_data/provenance/eccc_daily_download_log.csv')
JOBS = {
 3698:(1950,2008), 47407:(2008,2013), 51097:(2018,2023), 27174:(1996,2023), 28051:(1999,2023),
 3471:(1950,2012), 50821:(2012,2023), 3472:(1950,2010),
 3517:(1950,1971), 3519:(1970,2023), 3520:(1962,1996),
 3595:(1950,1997), 26963:(1997,2009), 48068:(2009,2023),
 3628:(1950,1987), 3629:(1954,1959), 3630:(1961,2007),
}
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
            with urllib.request.urlopen(req,timeout=90) as r:
                data=r.read()
            text=data.decode('utf-8-sig',errors='strict')
            # An annual daily response must contain a Date/Time header.
            if 'Date/Time' not in text[:2000]:
                raise ValueError('response lacks Date/Time header: '+text[:120].replace('\n',' '))
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_bytes(data)
            return (sid,year,'downloaded',len(data),url)
        except Exception as e:
            last=f'{type(e).__name__}: {e}'
            time.sleep(2*(attempt+1))
    return (sid,year,'FAILED',0,last)
jobs=[(sid,y) for sid,(a,b) in JOBS.items() for y in range(a,b+1)]
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    for i,r in enumerate(ex.map(fetch,jobs),1):
        rows.append(r)
        if i%50==0: print(f'{i}/{len(jobs)}',flush=True)
LOG.parent.mkdir(parents=True,exist_ok=True)
with LOG.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['station_id','year','status','bytes','url_or_error']); w.writerows(sorted(rows))
print('jobs',len(rows),'failed',sum(r[2]=='FAILED' for r in rows))
