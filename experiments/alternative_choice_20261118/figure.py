"""Publish the observed alternative actions and the response to the offered point advantage."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

root=Path(__file__).resolve().parent
for weight in ("Regular","Bold"):
    font_manager.fontManager.addfont(str(Path.home()/f".fonts/NotoSansCJKtc-{weight}.otf"))
plt.rcParams.update({"font.family":"Noto Sans CJK TC","font.size":11,"axes.unicode_minus":False})
a=json.loads((root/"results.json").read_text())["evaluation_mechanism"]
b=json.loads((root/"value_response.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(11.6,4.95),gridspec_kw={"width_ratios":[1.12,1.]})
fig.subplots_adjust(left=.135,right=.97,top=.76,bottom=.20,wspace=.40)
fig.text(.055,.925,"棋手如何使用原本可以直逃的一手",size=21,weight="bold",color="#172A32")
fig.text(.055,.85,"第 3、4 驗證折  ·  同一棋局的同一手只計一次",size=11.5,color="#485C65")
ax=axes[0]
cats=[("immediate_rescue","當手提子解圍"),("new_atari","另叫吃一串棋"),("other_far","其他遠處落子"),
      ("other_near","其他近處落子"),("capture","另處提子")]
for i,(key,name) in enumerate(cats):
    v=a[key]["share"];ax.barh(i,v,color="#18736F" if i==0 else "#7CA7A1",height=.56,zorder=2)
    ax.text(v+.014,i,f"{v:.1%}",va="center",size=10)
ax.set_title("43,260 次選擇別手，實際做了什麼？",loc="left",pad=13,size=12,weight="bold")
ax.set_yticks(range(5),[j[1] for j in cats]);ax.invert_yaxis();ax.set_xlim(0,.66)
ax.set_xticks([0,.2,.4,.6]);ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.grid(axis="x",color="#E4EAE9",zorder=0)
ax=axes[1]
groups=[("below_minus_1","少逾 1 目"),("within_1","相差 1 目內"),("above_1","多逾 1 目")]
for i,(key,_) in enumerate(groups):
    row=b["bins"][key];v=row["rescue_among_two"];n=row["rescue"]+row["direct"]
    ax.bar(i,v,width=.50,color=["#A45E4B","#779796","#18736F"][i],zorder=2)
    ax.text(i,v+.018,f"{v:.1%}",ha="center",size=10,weight="bold")
    ax.text(i,.052,f"{n:,} 次",ha="center",va="bottom",size=8.8,color="white")
ax.set_title("提子解圍相對直逃的估值與實戰採用率",loc="left",pad=13,size=12,weight="bold")
ax.set_xticks(range(3),[v for _,v in groups]);ax.set_ylabel("在兩手之中選提子解圍的比例")
ax.set_ylim(0,1.08);ax.set_yticks([0,.25,.5,.75,1]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
ax.grid(axis="y",color="#E4EAE9",zorder=0)
for ax in axes:
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0,pad=6,labelcolor="#334850")
fig.text(.055,.068,"右圖只計有唯一提子解圍選項，且實戰在它與直逃之間二選一的局面。目差是同一盤面兩條分支的引擎估值。",size=10,color="#485C65")
fig.savefig(root/"替代選擇與機會成本.png",dpi=160,facecolor="white")
plt.close(fig)
