import json
from pathlib import Path
M=Path('/home/u2507115/BK_AICUP_2026-1'); B=Path(__file__).parent/'branch'
js=lambda p:json.loads((M/p).read_text())
ERR=js('results/l1b/error_analysis.json')
print('models',ERR['models'],'per_player',ERR['per_player'],'questions',ERR['questions'])
def short(v,d=0):
    if isinstance(v,dict): return {k:short(x,d+1) for k,x in list(v.items())[:14]} if d<3 else '...'
    if isinstance(v,float): return round(v,4)
    if isinstance(v,list): return v[:6]
    return v
print('resid',json.dumps(short(ERR['residuals']),ensure_ascii=False)[:2200])
print('opp ch',json.dumps(short(ERR['opponent_channel']),ensure_ascii=False)[:700])
print('mlp_diff by',[k for k in ERR['scores']['mlp_diff']], 'all keys in mlp',list(ERR['scores']['mlp']))
FP=js('results/seq/fingerprint_check_v2.json')['same_n5']['base_tsfel16']; print('tsfel seen',{k:round(v,4) for k,v in FP.items() if isinstance(v,float)})
FS=js('results/seq/feature_stability.json')['summary']
for n in ('10','5'):
    print('tsfel pf',n,FS[n]['tsfel'].get('per_feature_corr'))
H=json.loads((B/'feature_stability_hd.json').read_text())['summary']
for n in ('10','5'):
    print('HD',n,{k:(short(v) ) for k,v in H[n].items() if 'hd' in k.lower()})
print('fp keys',list(js('results/seq/fingerprint_check_v2.json')))
d=js('results/l1b/stack_check3.json'); print('stack',{k:round(v['balanced_mean_over_n'],4) for k,v in d['scores'].items()})
