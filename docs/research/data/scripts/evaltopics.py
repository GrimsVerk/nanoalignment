import sys,collections,re,pyarrow.parquet as pq
d=sys.argv[1]
for s in ['article_questions','textbook_questions']:
    t=pq.read_table(f'{d}/{s}.parquet').to_pylist()
    c=collections.Counter(x for r in t for x in (r['topics'] or []))
    pat=re.compile(r'reward hack|specification gaming|test(s)? (pass|cases)|unit test|log(s|ging)?\b|monitor|oversight|sandbag|honest',re.I)
    hits=[r for r in t if pat.search(r['question']+' '.join(r['choices']))]
    rh=[r for r in t if re.search(r'reward hack|specification gaming|unit test|test case|delete.*log|log file',r['question']+' '.join(r['choices']),re.I)]
    print(s,len(t),'top topics',c.most_common(15))
    print('  broad-pattern hits',len(hits),' reward-hack/test/log hits',len(rh))
    for r in rh[:3]: print('   EX:',r['question'][:220])
    print('  misaligned_choice dist',collections.Counter(r['misaligned_choice'] for r in t))
