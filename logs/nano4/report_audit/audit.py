import json,sys,csv,collections
from pathlib import Path
M=Path('/home/u2507115/BK_AICUP_2026-1'); B=Path(sys.argv[1])
js=lambda p:json.loads((M/p).read_text())
def s(d,k): return d['scores'][k]['balanced_mean_over_n']
OPP,F0,ERR,HD,DIFF0,ST,LAD=js('results/l1b/opp_check.json'),js('results/l1b/opp_fold0_check.json'),js('results/l1b/error_analysis.json'),js('results/seq/hd_check.json'),js('results/models/rank_model_fused_opp_diff.json')['fold0_check'],js('results/l1b/stack_check3.json'),js('results/ladder/cv_summary.json')
print('score keys ERR mlp_diff',{k:v for k,v in ERR['scores']['mlp_diff'].items() if not isinstance(v,dict)})
print('DIFF0 keys',list(DIFF0)[:12],list(DIFF0['scores'])[:12])
for n,cv,f in [('hsl',s(OPP,'hsl'),s(F0,'hsl')),('fused',s(OPP,'fused'),s(F0,'fused_avg')),('fused_opp',s(OPP,'fused_opp'),s(F0,'fused_opp_avg')),('diff',s(ERR,'mlp_diff'),DIFF0['scores']['fused_opp_diff_avg']['balanced_mean_over_n'])]:
    print(f'{n:10s} CV {cv:.4f} f0 {f:.4f} f0-CV {f-cv:+.4f}')
print('HD',s(HD,'fused_opp_diff_hd'),s(HD,'fused_opp_diff'))
def walk(d,pfx=''):
    for k,v in d.items():
        if isinstance(v,dict) and 'diff' in v: print(' ',pfx+k,{x:(round(v[x],4) if isinstance(v[x],float) else v[x]) for x in v if x in('diff','ci95','folds_positive','per_fold','pass','passed','all_pass','conditions')})
        elif isinstance(v,dict): walk(v,pfx+k+'/')
for name,p in [('OPP','results/l1b/opp_check.json'),('F0','results/l1b/opp_fold0_check.json'),('ERR','results/l1b/error_analysis.json'),('HD','results/seq/hd_check.json'),('SE','results/l1b/se_check.json'),('Q','results/quality/quality_check.json'),('ORD','results/l1b/ordinal_check.json'),('XF','results/l1b/xfit_center_check.json'),('HIST','results/seq/hist_check.json'),('SHAPE','results/seq/shape_check.json'),('GB','results/gbdt/gbdt_check.json')]:
    print('==',name); walk(js(p).get('comparisons',{}))
print('== DIFF0 cmp',DIFF0.get('fused_opp_diff_avg_vs_fused_opp_avg'))
for name,f in [('TSFEL','tsfel_check.json'),('QMODEL','qmodel_check.json'),('MAMBA3','seq_check_mamba3.json')]:
    d=json.loads((B/f).read_text()); print('==',name,{k:round(v['balanced_mean_over_n'],4) for k,v in d['scores'].items()}); walk(d.get('comparisons',{}))
print('== FSHD',json.dumps(json.loads((B/'feature_stability_hd.json').read_text()).get('summary'),ensure_ascii=False)[:900])
print('== seq_check',{k:round(v['balanced_mean_over_n'],4) for k,v in js('results/seq/seq_check.json')['scores'].items()})
print('== ladder',{k:round(v['balanced_mean_over_n'],4) for k,v in LAD['rungs'].items()}, 'ens',round(s(ST,'ens_top3'),4))
FP=js('results/seq/fingerprint_check_v2.json')['same_n5']
for k,v in FP.items():
    if isinstance(v,dict): print('D5',k,{x:(round(y,4) if isinstance(y,float) else y) for x,y in v.items() if x in('gain_new_plain','gain_seen_plain','D_plain','D_plain_ci95','D_balanced','D_balanced_ci95')})
FS=js('results/seq/feature_stability.json')['summary']
for n in ('10','5'):
    print('FS',n,{k:(round(v['median_split_half_corr'],3),round(v['mean_ratio_W_over_between'],2)) for k,v in FS[n].items()})
V=js('results/viz_examples.json')['position']; print('viz',V['t'],V['tv_12k_6d'],V['played'],V['played_logp']['12k'],V['played_logp']['6d'])
print('perclass hsl',{k:round(v,3) for k,v in OPP['scores']['hsl']['per_class'].items()})
print('perclass fopp',{k:round(v,3) for k,v in OPP['scores']['fused_opp']['per_class'].items()})
print('ERR resid',{k:v for k,v in ERR['residuals'].items() if k in('between_player_share','all_right','all_wrong')})
print('ERR other',[k for k in ERR])
print('baseline',{k:v for k,v in js('results/baseline_official_eval.json').items() if not isinstance(v,(dict,list))}, list(js('results/baseline_official_eval.json'))[:8])
c=collections.Counter(); pl=set(); allp=set(); gm=0
with open(M/'results/data/hd_table.csv') as f:
    for r in csv.DictReader(f):
        gm+=1; allp.add(r['player_id'])
        h=float(r['hd']) if r['hd'] not in('','nan') else -1
        c[h]+=1
        if h>=2: pl.add(r['player_id'])
print('hd_table games',gm,'players',len(allp),'HD>=2 players',len(pl),'counts',sorted(c.items()))
