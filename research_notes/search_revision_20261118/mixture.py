"""Describe how four observed board conditions contribute to raw versus matched destination differences."""
import itertools
import json
import math
import numpy as np
from destination import HERE, data, distance, features, save_json, sha
from mechanism import matched_difference
from aicup.metric import LEVELS

NAMES=["phase","prior_mass","candidate_gap","last_opponent_distance"]
DIMS=[3,3,2,3]


def main():
    games,e=data()
    _,_,miss,near,encoded=features(e);gid=e["game"]
    labels=np.array(np.unravel_index(encoded,DIMS)).T
    previous=np.where(e["last"]==361,37,np.minimum(distance(e["last"],e["A"]),distance(e["last"],e["B"])))
    np.testing.assert_array_equal(labels[:,3],np.digitize(previous,[3.5,8.5]))
    results={}
    expected=json.loads((HERE/"destination.json").read_text())
    for tag,include in (("question_games",(games["fold"][gid]>=3)&~games["fresh"][gid]),("new_games",games["fresh"][gid])):
        selected=include&miss
        grades=games["y"][gid[selected]].astype(np.int64);y=near[selected];columns=labels[selected];counts={}
        for subset_len in range(5):
            for subset in itertools.combinations(range(4),subset_len):
                c=np.zeros(len(grades),int);ncell=1
                for j in subset:
                    c=c*DIMS[j]+columns[:,j];ncell*=DIMS[j]
                ns=np.bincount(grades*ncell+c,minlength=10*ncell).reshape(10,ncell)
                ss=np.bincount(grades*ncell+c,weights=y,minlength=10*ncell).reshape(10,ncell)
                counts[subset]=(ns,ss)
        by_pair={}
        for lo,hi in ((4,5),(6,7),(0,9)):
            contrast={s:float(matched_difference(n[lo],k[lo],n[hi],k[hi])) for s,(n,k) in counts.items()}
            parts={}
            for ix,name in enumerate(NAMES):
                part=0
                for subset in contrast:
                    if ix in subset:continue
                    t=len(subset);weight=math.factorial(t)*math.factorial(3-t)/math.factorial(4)
                    part+=weight*(contrast[subset]-contrast[tuple(sorted((*subset,ix)))])
                parts[name]=part
            all_axes=contrast[(0,1,2,3)];raw=contrast[()]
            assert np.isclose(raw-all_axes,sum(parts.values()))
            key=LEVELS[lo]+"/"+LEVELS[hi]
            assert np.isclose(all_axes,expected["samples"][tag]["matched_near_given_miss"][key]["difference"])
            by_pair[key]={"raw_difference":raw,"adjusted_difference":all_axes,"amount_removed":parts}
        distance_rows={}
        for g in range(10):
            n,s=counts[(3,)];distance_rows[LEVELS[g]]={"n":n[g].tolist(),"near":s[g].astype(int).tolist(),
                "exposure_share":(n[g]/n[g].sum()).tolist(),"near_rates":(s[g]/n[g]).tolist()}
        n,s=counts[(3,)];totals=n.sum(0);pooled=s.sum(0)/totals
        results[tag]={"contrasts":by_pair,"opponent_distance":distance_rows,"pooled_near_given_opponent_distance":pooled.tolist()}
    out={"status":"exploratory descriptive exact decomposition across all 24 orderings of four preselected board variables; no causal interpretation",
         "samples":results,"sha256":{p.name:sha(p) for p in (HERE/"destination.json",HERE/"mixture.py")}}
    save_json(HERE/"mixture.json",out)
    for tag,r in results.items():
        print(tag,r["contrasts"],r["pooled_near_given_opponent_distance"],flush=True)

if __name__=="__main__":main()
