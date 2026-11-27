"""Display the board-context effect against matched adjacent-rank contrasts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from interaction import new_data
from destination import features
from corpus import HERE,save_json

for weight in ("Regular","Bold"):
    font_manager.fontManager.addfont(str(Path.home()/f".fonts/NotoSansCJKtc-{weight}.otf"))
plt.rcParams.update({"font.family":"Noto Sans CJK TC","axes.unicode_minus":False,"font.size":10})
mix=json.loads((HERE/"mixture.json").read_text())["samples"]
dest=json.loads((HERE/"destination.json").read_text())["samples"]
g,e=new_data();_,_,miss,near,cell=features(e)
n=np.bincount(cell[miss]%3,minlength=3);k=np.bincount(cell[near]%3,minlength=3)
extra={"sample":"reserved_extra","games":len(g["keys"]),"misses":int(n.sum()),"denominators_near_mid_far":n.tolist(),
       "near_numerators":k.tolist(),"near_rates":(k/n).tolist()}
save_json(HERE/"reserved_position_rates.json",extra)
tag_names=["原題棋譜","第一批額外棋譜","另行保留的棋譜"];colors=["#1C615C","#B46547","#667D95"]
fig,(a,b)=plt.subplots(1,2,figsize=(10.9,4.5),gridspec_kw={"width_ratios":[1.17,1]})
fig.subplots_adjust(top=.76,left=.08,right=.97,bottom=.24,wspace=.37)
fig.text(.065,.91,"對手上一手的位置，比相鄰段位更能區分這一手會下在哪裡",size=16.5,weight="bold",color="#263734")
rates=np.array([mix["question_games"]["pooled_near_given_opponent_distance"],
      mix["new_games"]["pooled_near_given_opponent_distance"],extra["near_rates"]])*100
xx=np.arange(3)
for j,(name,col) in enumerate(zip(tag_names,colors)):
    pos=xx+(j-1)*.23
    a.bar(pos,rates[j],.20,color=col,zorder=3,label=name)
a.set_ylabel("另走附近的比例（%）");a.set_ylim(0,85);a.set_yticks([0,20,40,60,80])
a.set_xticks(xx,["至多 3 格","4 至 8 格","至少 9 格"]);a.set_xlabel("對手上一手到較近候選點的距離",labelpad=7)
a.set_title("兩個候選點都沒下，實戰還會留在附近嗎？",loc="left",weight="bold",pad=12,size=10.8)
a.legend(ncol=1,loc="upper right",frameon=False,fontsize=8)
a.grid(axis="y",color="#D9DFDF",linewidth=.7);a.spines[["top","right"]].set_visible(False)
for j,tag in enumerate(("question_games","new_games")):
    points=[];low=[];high=[]
    for pair in ("1D/2D","3D/4D"):
        row=dest[tag]["matched_near_given_miss"][pair];x=row["difference"]*100;ci=np.array(row["ci95"])*100
        points.append(x);low.append(x-ci[0]);high.append(ci[1]-x)
    b.errorbar(points,np.array([1.,0.])+(.095 if j==0 else -.095),xerr=[low,high],fmt="o" if j==0 else "D",
               color=colors[j],capsize=3,label=tag_names[j],ms=6,elinewidth=1.5)
b.axvline(0,c="#718580",lw=1);b.grid(axis="x",color="#D9DFDF",lw=.7)
b.set_yticks([1,0],["2D 減 1D","4D 減 3D"]);b.set_ylim(-.48,1.55)
b.set_xlabel("另走附近的機率差（百分點）",labelpad=7)
b.set_title("對齊盤面條件後，相鄰段位差多少？",loc="left",weight="bold",pad=12,size=10.8)
b.spines[["top","right","left"]].set_visible(False);b.tick_params(axis="y",length=0)
fig.text(.08,.065,"只取兩候選點相距至多 4 格、實戰另選落點的局面。附近指實戰距任一候選點至多 3 格。右圖橫線為按棋手重抽的 95% 區間。",fontsize=8.3,color="#4D5E5B")
fig.savefig(HERE/"對手上一手與落點.png",dpi=175,bbox_inches="tight",facecolor="white");plt.close(fig)
