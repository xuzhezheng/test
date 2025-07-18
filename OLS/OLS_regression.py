"""Run OLS regressions for ESG uncertainty analysis."""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from utils import merge_df, drop_high_vif


def first_stage_regression(
    y="ESG_Uncertainy_others2",
    x="senti",
    cvs=None,
):
    """Run OLS regressions with different outcome dimensions.

    Parameters
    ----------
    y : str
        Target variable identifier.
    x : str
        Feature set identifier.
    cvs : list[str] | None
        Control variables to include in the model.
    """
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

    # Load treatment / feature variable
    if x == "senti":
        x = pd.read_csv("../data/X/企业环保新闻senti.csv")
    elif x == "newsnum":
        x = pd.read_csv("../data/X/news_num.csv")
        x["Scode"] = x["Scode"].astype(int)
        x["Year"] = x["Year"].astype(int)
    x_colname = x.columns[-1]

    # Load control variables
    cv = pd.read_csv("../data/CV/control_variable3.csv")

    # Load dependent variable
    if y == "ESG_Uncertainty":
        y = pd.read_csv("../data/Y/ESG_Uncertainty.csv")
        y.columns = ["Scode", "Year", "ESG_Uncert"]
    elif y == "ESG_Uncertainy_others":
        y = pd.read_csv("../data/Y/ESG_Uncertainty_others.csv")
        y.columns = ["Scode", "Year", "ESG_Uncert"]
    elif y == "ESG_Uncertainy_others2":
        y = pd.read_csv("../data/Y/ESG_Uncertain_others2.csv")
        y.columns = [
            "Scode",
            "Year",
            "ESG_Uncert",
            "E_Uncert",
            "S_Uncert",
            "G_Uncert",
        ]

    df = merge_df(y, x, cv)
    cvs = drop_high_vif(df, cvs, threshold=10)

    firm_counts = df["Scode"].value_counts()
    selected_firms = firm_counts[firm_counts >= 4].index
    df = df[df["Scode"].isin(selected_firms)]

    # Remove extreme industry
    df = df[df["indcd"] != 17]

    df_ESG = df.drop(columns=["E_Uncert", "S_Uncert", "G_Uncert"]).dropna()
    formula = (
        "ESG_Uncert ~ 1 + "
        + x_colname
        + " + "
        + " + ".join(cvs)
        + " + C(Year) + C(indcd) + C(Scode)"
    )
    model_stage1 = smf.ols(formula=formula, data=df_ESG).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_ESG["indcd"]},
    )
    print("=====" * 10, "model_stage1", "=====" * 10)
    print(model_stage1.summary())

    df_E = df.drop(columns=["ESG_Uncert", "S_Uncert", "G_Uncert"]).dropna()
    formula = (
        "E_Uncert ~ 1 + "
        + x_colname
        + " + "
        + " + ".join(cvs)
        + " + C(Year) + C(indcd) + C(Scode)"
    )
    model_stage2 = smf.ols(formula=formula, data=df_E).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_E["indcd"]},
    )
    print("=====" * 10, "model_stage2", "=====" * 10)
    print(model_stage2.summary())

    df_S = df.drop(columns=["ESG_Uncert", "E_Uncert", "G_Uncert"]).dropna()
    formula = (
        "S_Uncert ~ 1 + "
        + x_colname
        + " + "
        + " + ".join(cvs)
        + " + C(Year) + C(indcd) + C(Scode)"
    )
    model_stage3 = smf.ols(formula=formula, data=df_S).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_S["indcd"]},
    )
    print("=====" * 10, "model_stage3", "=====" * 10)
    print(model_stage3.summary())

    df_G = df.drop(columns=["ESG_Uncert", "E_Uncert", "S_Uncert"]).dropna()
    formula = (
        "G_Uncert ~ 1 + "
        + x_colname
        + " + "
        + " + ".join(cvs)
        + " + C(Year) + C(indcd) + C(Scode)"
    )
    model_stage4 = smf.ols(formula=formula, data=df_G).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_G["indcd"]},
    )
    print("=====" * 10, "model_stage4", "=====" * 10)
    print(model_stage4.summary())


if __name__ == "__main__":
    first_stage_regression(
        y="ESG_Uncertainy_others2",
        x="newsnum",
        cvs=[
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
            "Dig_Level",
            "Age",
            "Boardsize",
        ],
    )
