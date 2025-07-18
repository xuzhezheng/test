"""DMLIV estimation using tree-based models."""

import numpy as np
import pandas as pd
from econml.iv.dml import DMLIV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats.mstats import winsorize

from utils import merge_df


def dmliv_tree(
    y: str = "ESG_Uncertainy_others2",
    x: str = "senti",
    ivs: str = "network_mean",
    cvs: list[str] | None = None,
) -> None:
    """Run DMLIV using random forests."""
    if cvs is None:
        cvs = [
            "Inst_invest",
            "Inde_director",
            "Ten_share",
            "Duality",
            "Analyst",
            "Assets",
            "Debt",
            "ROA",
            "Fixed",
            "Bktomk",
        ]

    if x == "senti":
        x = pd.read_csv("../data/X/企业环保新闻senti.csv")
    elif x == "newsnum":
        x = pd.read_csv("../data/X/news_num.csv")
        x["Scode"] = x["Scode"].astype(int)
        x["Year"] = x["Year"].astype(int)
    else:
        x = pd.read_csv("./x/cnt_poscnt_negcnt_maxsenti_minsenti_meansenti_stdse" "nti.csv")
    x_col = x.columns[-1]

    df6 = pd.read_csv("../data/IV/企业环保新闻count_b3.csv")
    df6 = df6[["Scode", "Year", "newsnum_3yr_avg"]]
    df6["newsnum_3yr_avg"] = np.log1p(df6["newsnum_3yr_avg"])

    df7 = pd.read_csv("../data/IV/企业环保新闻count-同行业同省平均值.csv")
    df7 = df7[["Scode", "Year", "peer_avg_news_count"]]
    df7["peer_avg_news_count"] = np.log1p(df7["peer_avg_news_count"])

    iv = merge_df(df6, df7)
    iv_list = ["newsnum_3yr_avg", "peer_avg_news_count"]

    if ivs == "network_mean":
        df5 = pd.read_csv("../data/IV/上市公司高管网络新闻_mean.csv")
        df5.columns = ["Scode", "Year", "G_news"]
        iv = merge_df(iv, df5)
        iv_list.append(df5.columns[-1])
    elif ivs == "network_count":
        df5 = pd.read_csv("../data/IV/上市公司高管网络新闻_count.csv")
        df5.columns = ["Scode", "Year", "G_news"]
        iv = merge_df(iv, df5)
        iv_list.append(df5.columns[-1])
    elif ivs == "paper_mean":
        df5 = pd.read_csv("../data/IV/上市公司高管报刊新闻_mean.csv")
        df5.columns = ["Scode", "Year", "G_news"]
        iv = merge_df(iv, df5)
        iv_list.append(df5.columns[-1])
    elif ivs == "paper_count":
        df5 = pd.read_csv("../data/IV/上市公司高管报刊新闻_count.csv")
        df5[df5.columns[-1]] = np.log1p(df5[df5.columns[-1]])
        df5.columns = ["Scode", "Year", "G_news"]
        df5["G_news"] = winsorize(df5["G_news"], limits=[0.01, 0.01])
        iv = merge_df(iv, df5)
        iv_list.append(df5.columns[-1])

    cv = pd.read_csv("../data/CV/control_variable3.csv")

    if y == "ESG_Uncertainty":
        y = pd.read_csv("../data/Y/ESG_Uncertainty.csv")
    elif y == "ESG_Uncertainy_others":
        y = pd.read_csv("../data/Y/ESG_Uncertainty_others.csv")
    elif y == "ESG_Uncertainy_others2":
        y = pd.read_csv("../data/Y/ESG_Uncertain_others2.csv")
    y.columns = ["Scode", "Year", "ESG_Uncert"]
    y = y.dropna()
    y_col = y.columns[-1]

    df = merge_df(y, x, iv, cv)
    df = df[["Scode", "Year", y_col, x_col, *iv_list, *cvs]].dropna()

    X = df[cvs].values
    Y = df[y_col].values
    Z = df[iv_list].values
    T = winsorize(df[x_col].values, limits=[0.01, 0.01])

    est = DMLIV(
        model_y_xw=RandomForestRegressor(random_state=42),
        model_t_xwz=RandomForestRegressor(random_state=42),
        model_t_xw=RandomForestRegressor(random_state=42),
        model_final=RandomForestRegressor(random_state=42),
        discrete_treatment=False,
        cv=5,
    )
    est.fit(Y=Y, T=T, X=X, Z=Z)

    T_pred = est.models_t_xwz[0][0].predict(np.concatenate([X, Z], axis=1))
    r2 = r2_score(T, T_pred)
    rmse = mean_squared_error(T, T_pred, squared=False)
    print("Treatment R2:", r2)
    print("Treatment RMSE:", rmse)


if __name__ == "__main__":
    dmliv_tree(x="newsnum", ivs="paper_count")
