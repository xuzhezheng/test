"""Estimate the first stage of a 2SLS regression."""

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
from scipy.stats.mstats import winsorize

from utils import merge_df, drop_high_vif


def first_stage_iv(
    y: str = "ESG_Uncertainy_others2",
    x: str = "senti",
    ivs: str = "network_mean",
    cvs: list[str] | None = None,
) -> None:
    """Run the first-stage regression and print the summary."""
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
    x_colname = x.columns[-1]

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
    y.columns = ["Scode", "Year", "ESG_Uncert", "E_Uncert", "S_Uncert", "G_Uncert"]
    y = y.dropna()
    y_colname = y.columns[-4]

    df = merge_df(y, x, iv, cv)
    df = df[[
        "Scode",
        "Year",
        y_colname,
        x_colname,
        "indcd",
        "province",
        *iv_list,
        *cvs,
    ]]
    cvs = drop_high_vif(df, cvs, threshold=10)
    df = df.dropna()

    formula = (
        "ESG_Uncert ~ 1 + "
        + " + ".join(cvs)
        + f" [{x_colname} ~ "
        + " + ".join(iv_list)
        + "] + C(Year) + C(indcd)"
    )
    model = IV2SLS.from_formula(formula, data=df).fit(cov_type="clustered", clusters=df["indcd"])
    print(model.first_stage.summary)


if __name__ == "__main__":
    first_stage_iv(x="newsnum", ivs="paper_mean")
