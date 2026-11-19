import json,sys,urllib.request
for rid in sys.argv[1:]:
    kind='datasets' if not rid.startswith('M:') else 'models'
    rid=rid.removeprefix('M:')
    try:
        d=json.load(urllib.request.urlopen(f'https://huggingface.co/api/{kind}/{rid}',timeout=30))
    except Exception as e:
        print('==',rid,'ERR',e); continue
    cd=d.get('cardData') or {}
    tags=[t for t in d.get('tags',[]) if t.startswith(('license','size_categories','task_categories'))]
    print('==',rid,'| gated:',d.get('gated'),'| private:',d.get('private'),'| lic:',cd.get('license'),'|',tags,'| dl:',d.get('downloads'),'| modified:',str(d.get('lastModified'))[:10])
    if kind=='datasets':
        try:
            s=json.load(urllib.request.urlopen(f'https://datasets-server.huggingface.co/size?dataset={rid}',timeout=30))
            for c in s['size'].get('configs',[]): print('   cfg',c['config'],'rows',c.get('num_rows'),'bytes',c.get('num_bytes_parquet_files'))
            for c in s['size'].get('splits',[])[:12]: print('   split',c['config'],c['split'],'rows',c.get('num_rows'))
        except Exception as e: print('   size ERR',e)
    else:
        print('   params', (d.get('safetensors') or {}).get('total'), 'base', cd.get('base_model'))
