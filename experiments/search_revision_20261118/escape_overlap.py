"""Bridge pre-action whole-board menus to validated ladder escapes from the earlier experiment."""
import json
import numpy as np

from response import HERE, PREV, SHARDS, events, save_json, sha, strong
from search import COORD

ALTERNATIVE = HERE.parent / "alternative_choice_20261118"
KINDS = ["immediate_rescue","capture","new_atari","other_near","other_far","pass"]


def keys(game, ply):
    assert np.max(ply) < 65536
    return game.astype(np.int64)*65536 + ply


def main():
    games,e,_,_=events()
    with np.load(PREV/"game_counts.npz") as a:
        np.testing.assert_array_equal(a["keys"],games["keys"][~games["fresh"]])
    keep=(games["fold"][e["game"]]>=3)&~games["fresh"][e["game"]]
    q=np.flatnonzero(keep)
    menu_keys=keys(e["game"][q],e["ply"][q])
    offers=ALTERNATIVE/"offer_events.npz"; others=ALTERNATIVE/"events.npz"
    for p,log in ((offers,"offer_extraction.json"),(others,"extraction.json")):
        assert sha(p)==json.loads((ALTERNATIVE/log).read_text())["sha256"][p.name]
    with np.load(offers) as z:
        ok=games["fold"][z["game"]]>=3
        old={k:z[k][ok] for k in z.files}
    offered_keys=keys(old["game"],old["ply"])
    assert len(set(offered_keys))==len(offered_keys)
    _, current, past=np.intersect1d(menu_keys,offered_keys,assume_unique=True,return_indices=True)
    row=q[current];old={k:v[past] for k,v in old.items()}
    np.testing.assert_array_equal(e["actual"][row],old["actual"])
    a=e["A"][row];b=e["B"][row];escape=old["escape"]; actual=e["actual"][row]
    escape_a=escape==a
    has_escape=escape_a|(b==escape)
    alternative=np.where(escape_a,b,a)
    side=np.where(escape_a,1,-1)
    raw=side*e["d0"][row];searched=side*e["d24"][row]
    reversal=strong(raw,searched)
    picked_alt=actual==alternative
    picked_escape=actual==escape

    def stats(mask):
        other=mask & ~picked_escape & ~picked_alt
        named=(picked_escape|picked_alt)&mask
        return {"opportunities":int(mask.sum()), "used_direct":int((picked_escape&mask).sum()),
                "used_alternative":int((picked_alt&mask).sum()),"outside":int(other.sum()),
                "took_alternative_given_named":float((picked_alt&mask).sum()/named.sum()) if named.any() else None}

    all_sampled=[]
    deep_results={}
    for s in range(SHARDS):
        with np.load(HERE/f"root_{s}.npz") as z:
            k=(games["fold"][z["game"]]>=3)&~games["fresh"][z["game"]]
            all_sampled.append(keys(z["game"][k],z["ply"][k]))
        with np.load(HERE/f"search_{s}.npz") as z:
            deep_results.update(zip(z["event"][z["audit"]], z["s128A"]-z["s128B"]))

    with np.load(others) as z:
        recorded=keys(z["game"],z["ply"]);by_kind=dict(zip(recorded,z["kind"]))
        target_by_key=dict(zip(recorded,z["target"]))
        lost_by_key=dict(zip(recorded,z["lost_first"]))
    bridge_keys=menu_keys[current]
    ranked=has_escape&reversal
    alternate_results={}
    for name,mask in (("search_favors_direct",ranked&(searched<0)),
                      ("search_favors_alternative",ranked&(searched>0))):
        by_action=np.bincount([by_kind[k] for k in bridge_keys[mask&picked_alt]],minlength=len(KINDS))
        alternate_results[name]={**stats(mask),"played_alternative_kinds":dict(zip(KINDS,map(int,by_action)))}
    events_id=e["event"][row]
    d128=np.array([deep_results.get(i,np.nan) for i in events_id])*side
    good=ranked&picked_alt&(searched>0)&(d128>=.5)
    pick=np.flatnonzero(good)
    example=None
    if len(pick):
        i=pick[np.argmin(events_id[pick])]
        example={"event":int(events_id[i]),"game":int(e["game"][row[i]]),"ply":int(e["ply"][row[i]]),
            "mover":"W" if old["white"][i] else "B","target_point":int(target_by_key[bridge_keys[i]]),
            "escape":int(escape[i]),"alternative":int(alternative[i]),"actual":int(actual[i]),
            "alternative_gtp":COORD[alternative[i]],"escape_gtp":COORD[escape[i]],"target_size":int(old["size"][i]),
            "alternative_kind":KINDS[by_kind[bridge_keys[i]]], "lost_on_opponents_first_reply":int(lost_by_key[bridge_keys[i]]),
            "raw_alt_minus_direct":float(raw[i]),"v24_alt_minus_direct":float(searched[i]),"v128_alt_minus_direct":float(d128[i]),
            "selection":"lowest event ID with an audited reversal away from direct and the alternative played"}
    out={"earlier_eligible_escapes":len(offered_keys),"scheduled_escapes":len(np.intersect1d(offered_keys,np.concatenate(all_sampled),assume_unique=True)),
         "eligible_two_move_menus_on_escape":len(row),"escape_among_top_two":stats(has_escape),"strong_reversals":stats(ranked),
         "strong_by_direction":alternate_results,"audited_examples_meeting_rule":len(pick),"example":example,
         "sha256":{p.name:sha(p) for p in (offers,others,HERE/"escape_overlap.py",HERE/"protocol.json")}}
    save_json(HERE/"escape_overlap.json",out)
    print(json.dumps({k:v for k,v in out.items() if k!="sha256"},indent=2),flush=True)


if __name__=="__main__":
    main()
