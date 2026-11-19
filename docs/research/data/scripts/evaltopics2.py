import sys,re,pyarrow.parquet as pq
d=sys.argv[1]
pat=re.compile(r'hack|gaming|specification|tamper|shortcut|cheat|test',re.I)
for s in ['article_questions','textbook_questions']:
    t=pq.read_table(f'{d}/{s}.parquet').to_pylist()
    sel=[r for r in t if any(pat.search(x) for x in (r['topics'] or []))]
    print(s,'questions with hack/gaming/tamper/test-like topics:',len(sel),'of',len(t))
    import statistics; print('  median question chars',statistics.median(len(r['question']) for r in t),' median choice chars',statistics.median(len(c) for r in t for c in r['choices']))
