"""Draw an audited real decision and the player-bootstrap contrasts for the target grades."""
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import numpy as np

from corpus import HERE, R1, SHARDS

for weight in ("Regular","Bold"):
    font_manager.fontManager.addfont(str(Path.home()/f".fonts/NotoSansCJKtc-{weight}.otf"))
plt.rcParams.update({"font.family":"Noto Sans CJK TC","font.size":10,"axes.unicode_minus":False})
TEAL="#17675E"; RUST="#AA553B"; INK="#222F31"; GRID="#D7DFDF"


def draw_board(ax, sample):
    sys.path.insert(0,str(R1/"repo/KataGo/python"))
    from katago.game.board import Board
    g,ply=sample["game"],sample["ply"]
    with gzip.open(HERE/f"corpus_{g%SHARDS}.tsv.gz","rt") as f:
        line=next(l for l in f if l.split("\t",1)[0]==str(g))
    _,_,who,_,setup,seq=line.rstrip().split("\t")
    assert who==sample["mover"]
    b=Board(19)
    def loc(x,y):
        return Board.PASS_LOC if x=="-" else b.loc(ord(x)-97,ord(y)-97)
    if setup!="-":
        for i in range(0,len(setup),2):
            b.set_stone(Board.BLACK,loc(*setup[i:i+2]))
    for i in range(ply):
        c,x,y=seq[3*i:3*i+3]
        b.play(Board.BLACK if c=="B" else Board.WHITE,loc(x,y))
    c,x,y=seq[3*ply:3*ply+3]
    assert c==who and (ord(y)-97)*19+ord(x)-97==sample["actual"]
    side=Board.BLACK if who=="B" else Board.WHITE
    e,a,t=sample["escape"],sample["alternative"],sample["target_point"]
    idx=lambda v:b.loc(v%19,v//19)
    assert b.board[idx(t)]==side and b.num_liberties(idx(t))==1
    assert b.would_be_legal(side,idx(e)) and b.would_be_legal(side,idx(a))
    group=b.group_head[idx(t)]
    ax.set_aspect("equal");ax.set_facecolor("#F3E5C6")
    for j in range(19):
        ax.plot([0,18],[j,j],color="#9B8C70",lw=.55,zorder=1)
        ax.plot([j,j],[0,18],color="#9B8C70",lw=.55,zorder=1)
    for j in (3,9,15):
        for k in (3,9,15):ax.scatter(j,k,s=10,color="#61563F",zorder=2)
    for k in range(361):
        color=b.board[idx(k)]
        if color not in (Board.BLACK,Board.WHITE):continue
        x,y=k%19,k//19
        ax.add_patch(plt.Circle((x,y),.44,facecolor="#293136" if color==Board.BLACK else "#FAFAF6",
                                edgecolor="#323A3B",linewidth=.7,zorder=3))
        if b.group_head[idx(k)]==group:
            ax.scatter(x,y,marker="s",s=13,facecolors="none",edgecolors="#CC6848",linewidth=.95,zorder=4)
    for k,marker,color in ((e,"o",TEAL),(a,"D",RUST)):
        ax.scatter(k%19,k//19,s=60,marker=marker,facecolors="#F8F3E6",edgecolors=color,lw=1.8,zorder=5)
    inds=[0,3,6,9,12,15,18];letters="ABCDEFGHJKLMNOPQRST"
    ax.set_xticks(inds,[letters[i] for i in inds]);ax.set_yticks(inds,[str(19-i) for i in inds])
    ax.set_xlim(-.65,18.65);ax.set_ylim(18.65,-.65);ax.tick_params(length=0,labelsize=8)
    for sp in ax.spines.values():sp.set_visible(False)


def main():
    mechanism=json.loads((HERE/"mechanism.json").read_text())
    sample=json.loads((HERE/"escape_overlap.json").read_text())["example"]
    fig,(left,right)=plt.subplots(1,2,figsize=(11.0,5.4),gridspec_kw={"width_ratios":[1,1.23]})
    fig.subplots_adjust(left=.07,right=.97,top=.79,bottom=.21,wspace=.36)
    fig.text(.065,.944,"當搜尋改變兩手的優劣，棋手選了哪一手？",color=INK,size=19,weight="bold")
    if sample:
        draw_board(left,sample)
        side="黑" if sample["mover"]=="B" else "白"
        left.set_title(f"{side}第 {sample['ply']+1} 手之前",loc="left",pad=10,weight="bold",color=INK)
        left.text(0,-.13,f"○ 直逃 {sample['escape_gtp']}　◇ 實戰 {sample['alternative_gtp']}　□ 被叫吃的棋",transform=left.transAxes,fontsize=9,color=INK)
        left.text(0,-.22,f"實戰手較直逃多出的目數　初估 {sample['raw_alt_minus_direct']:+.1f}　24 次 {sample['v24_alt_minus_direct']:+.1f}　128 次 {sample['v128_alt_minus_direct']:+.1f}",
                  transform=left.transAxes,fontsize=8.3,color=INK)
    else:
        sizes=["5","10","20"];v=[mechanism["question_coverage"][s]["strong_named_at_least_1"]*100 for s in sizes]
        left.bar(sizes,v,color=TEAL,width=.52,zorder=3);left.set_ylim(0,108);left.set_yticks(range(0,101,20));left.set_ylabel("至少出現一次的題目（%）")
        left.spines[["top","right"]].set_visible(False);left.grid(axis="y",color=GRID,linewidth=.7)
        left.set_xlabel("每題棋譜局數");left.set_title("多少題能看到翻轉後的選擇？",loc="left",weight="bold")
        for i,m in enumerate(v):left.text(i,m+2,f"{m:.1f}%",ha="center")
    pairs=[("1D/2D","2D 減 1D"),("3D/4D","4D 減 3D")]
    for j,(tag,label,color,marker) in enumerate((("question_games","原題棋譜",TEAL,"o"),("new_games","同批棋手的額外棋譜",RUST,"D"))):
        x=[];lo=[];hi=[]
        for k,_ in pairs:
            row=mechanism["samples"][tag]["matched_strong_reversal"][k]
            point=100*row["difference"];ci=100*np.array(row["ci95"])
            x.append(point);lo.append(point-ci[0]);hi.append(ci[1]-point)
        yy=np.array([1.,0.])+(.10 if j==0 else -.10)
        right.errorbar(x,yy,xerr=[lo,hi],fmt=marker,elinewidth=1.5,capsize=3,ms=6,color=color,label=label,zorder=3)
    right.axvline(0,color="#718081",lw=1);right.set_ylim(-.5,1.7)
    right.set_yticks([1,0],[p[1] for p in pairs]);right.set_xlabel("選搜尋較佳手的機率差（百分點）",labelpad=9)
    right.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:+g}" if v else "0"))
    right.grid(axis="x",color=GRID,linewidth=.7);right.spines[["left","top","right"]].set_visible(False)
    right.tick_params(axis="y",length=0)
    right.set_title("相似局面中，相鄰段位差多少？",loc="left",weight="bold",color=INK)
    right.legend(loc="upper left",bbox_to_anchor=(0,1),fontsize=9,frameon=False,ncol=1)
    right.text(0,-.28,"兩次估值優劣翻轉且都差至少半目，實戰走其中一手。\n依候選順位、落子階段、搜尋方向及目差分層，橫線為按棋手重抽的 95% 區間。",
               transform=right.transAxes,fontsize=8.5,color=INK,linespacing=1.55)
    fig.savefig(HERE/"搜尋翻轉與段位.png",dpi=175,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    main()
