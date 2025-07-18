import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def merge_df(*dfs):
    """Merge multiple DataFrames on ``Scode`` and ``Year`` columns."""
    result = dfs[0]
    for df in dfs[1:]:
        result = pd.merge(result, df, on=["Scode", "Year"], how="inner")
    return result


def drop_high_vif(df, columns, threshold=10.0):
    """Return variables whose VIF values are below ``threshold``."""
    X = df[columns].dropna().astype(float)
    X = sm.add_constant(X)

    vif = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=["const"] + columns,
    )
    print("\nVIF 检查：")
    print(vif)

    selected = [var for var, val in vif.items() if val < threshold and var != "const"]
    return selected
