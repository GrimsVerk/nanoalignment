import json,sys,urllib.request,statistics
ds,cfg,split=sys.argv[1:4]
u=f'https://datasets-server.huggingface.co/first-rows?dataset={ds}&config={cfg}&split={split}'
try: d=json.load(urllib.request.urlopen(u,timeout=60))
except Exception as e: print(ds,'ERR',e); sys.exit()
rows=d.get('rows',[]); tr=sum(1 for r in rows if r.get('truncated_cells'))
lens=[len(json.dumps(r['row'])) for r in rows]
print(f"{ds}/{cfg}/{split}: n={len(rows)} truncated={tr} median_chars={statistics.median(lens) if lens else 0:.0f} max={max(lens) if lens else 0} cols={list(rows[0]['row'].keys()) if rows else []}")
if len(sys.argv)>4 and rows: print('  SAMPLE:', json.dumps(rows[0]['row'])[:int(sys.argv[4])])
