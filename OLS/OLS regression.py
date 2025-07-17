import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def merge_df(y,x,cv):
    merged_df = pd.merge(y, x, on=['Scode', 'Year'], how='inner')
    merged_df = pd.merge(merged_df, cv, on=['Scode', 'Year'], how='inner')
    return merged_df
def drop_high_vif(df, columns, threshold=10):
    X = df[columns].dropna().astype(float)
    X = sm.add_constant(X)

    vif = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=['const'] + columns
    )
    print("\nVIF 检查：")
    print(vif)

    # 筛选低于阈值的变量
    selected = vif[vif < threshold].index.tolist()
    selected = [var for var in selected if var != 'const']
    return selected


def first_stage_regression(y='ESG_Uncertainy_others2',x='senti',cvs=['Inst_invest','Inde_director','Ten_share','Duality','Analyst','Assets','Debt','ROA','Fixed','Bktomk']):
    #X有两个，news_num和meansenti
    #Y有三个，ESG_Uncertainty,ESG_Uncertainy_others,ESG_Uncertainy_others2
    #CV：['Inst_invest','Inde_director','Ten_share','Duality',
    #                 'Analyst','Assets','Debt','ROA','Fixed','Bktomk']
    if x == 'senti':
        x = pd.read_csv('../data/X/企业环保新闻senti.csv')
    elif x =='newsnum':
        x = pd.read_csv("../data/X/news_num.csv")
        x['Scode'] = x['Scode'].astype(int)
        x['Year'] = x['Year'].astype(int)
    x_colname = x.columns[-1]
    x['news_count'] = x['news_count']

    #cv
    # cv = pd.read_csv("../data/CV/control_variable2.csv")
    cv = pd.read_csv("../data/CV/control_variable3.csv")

    #y
    if y == 'ESG_Uncertainty':
        y = pd.read_csv("../data/Y/ESG_Uncertainty.csv")
        y.columns = ['Scode', 'Year', 'ESG_Uncert']
    elif y == 'ESG_Uncertainy_others':
        y = pd.read_csv("../data/Y/ESG_Uncertainty_others.csv")
        y.columns = ['Scode', 'Year', 'ESG_Uncert']
    elif y == 'ESG_Uncertainy_others2':
        y = pd.read_csv("../data/Y/ESG_Uncertain_others2.csv")
        y.columns = ['Scode', 'Year', 'ESG_Uncert','E_Uncert','S_Uncert','G_Uncert']


    merged_df = merge_df(y,x,cv)

    df = merged_df
    cvs = drop_high_vif(df, cvs, threshold=10)
    firm_counts = df['Scode'].value_counts()
    selected_firms = firm_counts[firm_counts >= 4].index
    df = df[df['Scode'].isin(selected_firms)]




    #回归

    # CV = df[['Inst_invest','Inde_director','Ten_share','Duality',
    #                 'Analyst','Assets','Debt','ROA','Fixed','Bktomk']]#全部的控制变量
    # Y = df['ESG_Uncert']
    # x = df[['news_count','Inst_invest','Inde_director','Ten_share','Duality',
    #                 'Analyst','Assets','Debt','ROA','Fixed','Bktomk']]

    # X = df[[x_colname]+cvs]#筛选出显著的X

    # 什么都不考虑

    # formula = 'ESG_Uncert ~ 1 +' + x_colname + " + " + " + ".join(cvs)
    # model_stage1 = smf.ols(formula=formula, data=df).fit()





    #不加固定效应
    # formula = 'ESG_Uncert ~ ' + x_colname + " + " + " + ".join(cvs)
    # model_stage1 = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['indcd']})

    #加固定效应

    # formula = 'ESG_Uncert ~ ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year)'
    # model_stage1 = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['indcd']})

    # formula = 'ESG_Uncert ~ 1 + ' + x_colname  + ' + C(Year) + C(indcd)'
    # model_stage1 = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['indcd']})

    df_ESG = df.drop(columns=['E_Uncert', 'S_Uncert', 'G_Uncert'])
    df_ESG = df_ESG.dropna()
    df = df[df['indcd'] != 17]
    formula = 'ESG_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd) + C(Scode)'
    model_stage1 = smf.ols(formula=formula, data=df_ESG).fit(cov_type='cluster', cov_kwds={'groups': df_ESG['indcd']})

    # formula = 'ESG_Uncert ~ ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd) + C(province)'
    # model_stage1 = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['indcd']})
    print("====="*10,"model_stage1","====="*10)
    print(model_stage1.summary())

    df_E = df.drop(columns=['ESG_Uncert', 'S_Uncert', 'G_Uncert'])
    df_E = df_E.dropna()
    formula = 'E_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd) + C(Scode)'
    model_stage2 = smf.ols(formula=formula, data=df_E).fit(cov_type='cluster', cov_kwds={'groups': df_E['indcd']})
    print("=====" * 10, "model_stage2", "=====" * 10)
    print(model_stage2.summary())

    df_S = df.drop(columns=['ESG_Uncert', 'E_Uncert', 'G_Uncert'])
    df_S = df_S.dropna()
    formula = 'S_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd) + C(Scode)'
    model_stage3 = smf.ols(formula=formula, data=df_S).fit(cov_type='cluster', cov_kwds={'groups': df_S['indcd']})
    print("=====" * 10, "model_stage3", "=====" * 10)
    print(model_stage3.summary())

    df_G = df.drop(columns=['ESG_Uncert', 'E_Uncert', 'S_Uncert'])
    df_G = df_G.dropna()
    formula = 'G_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd) + C(Scode)'
    model_stage4 = smf.ols(formula=formula, data=df_G).fit(cov_type='cluster', cov_kwds={'groups': df_G['indcd']})
    print("=====" * 10, "model_stage4", "=====" * 10)
    print(model_stage4.summary())


#X有两个，news_num和meansenti
    #Y有三个，ESG_Uncertainty,ESG_Uncertainy_others,ESG_Uncertainy_others2
    #CV：['Inst_invest','Inde_director','Ten_share','Duality','Analyst',
                            # 'Assets','Debt','ROA','Fixed','Bktomk','Dig_Level','Age','Boardsize']
#之后换x,换y，换cv，就直接调整原始数据和这行命令就行，不用再换了头疼。

first_stage_regression(y='ESG_Uncertainy_others2',x='newsnum',cvs=['Inst_invest','Inde_director','Ten_share','Duality','Analyst',
                            'Assets','Debt','ROA','Fixed','Bktomk','Dig_Level','Age','Boardsize'])



