import glob, pandas as pd
df = pd.concat([pd.read_csv(p) for p in glob.glob("results/quality_smoke/fold1/quality_s0_p*.csv")])
order = ["D","C","B","A","1D","2D","3D","4D","5D","6D"]
print(df[["t_n","t_loss_mean","t_loss_median","t_gt3","t_wloss_mean","o_loss_mean","t_end_lead"]].describe().round(3).to_string())
g = df.groupby("rank")[["t_loss_mean","t_gt3","t_gt8","t_wloss_mean","o_loss_mean"]].mean().reindex(order)
print(g.round(3).to_string())
print("spearman rank idx vs t_loss_mean:", round(pd.Series([order.index(r) for r in df["rank"]]).corr(df["t_loss_mean"].reset_index(drop=True), method="spearman"), 3))
