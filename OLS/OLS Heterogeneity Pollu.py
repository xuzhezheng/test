import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

def merge_df(y,x,cv,hv):
    merged_df = pd.merge(y, x, on=['Scode', 'Year'], how='inner')
    merged_df = pd.merge(merged_df, cv, on=['Scode', 'Year'], how='inner')
    merged_df = pd.merge(merged_df,hv,on=['Scode'], how='inner')
    return merged_df
def draw_fig(df):
    # 设置图形风格
    sns.set(style='whitegrid')

    # 创建图形
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Pollution'], kde=True, bins=30, color='skyblue', edgecolor='black')

    # 添加均值线
    mean_score = df['Pollution'].mean()
    plt.axvline(mean_score, color='red', linestyle='--', label=f'Mean = {mean_score:.2f}')

    # 图形标签
    plt.title("Distribution of Greenwashing Score", fontsize=14)
    plt.xlabel("Greenwashing Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


def first_stage_regression(y='ESG_Uncertainy_others2',x='senti',Ind='CIE',cvs=['Inst_invest','Inde_director','Ten_share','Duality','Analyst','Assets','Debt','ROA','Fixed','Bktomk']):
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
    x['news_count'] = np.log(x['news_count'] + 1)

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

    if Ind == 'CIE':
        Ind = pd.read_csv("../data/Heterogeneity/Pollution & Non Indust CIE.csv")#2015-2020
        Ind = Ind[['Scode', 'Pollution']]

    elif Ind == 'ours':
        Ind = pd.read_csv("../data/Heterogeneity/Pollution & Non Indust ours.csv")#2015-2020


    # Ind = Ind[['Scode', 'Year', 'Pollution']]
    #Polluton = 0非污染, Pollution = 1 污染
    Ind = Ind.dropna()
    Ind['Scode'] = Ind['Scode'].astype(int)
    # Ind['Year'] = Ind['Year'].astype(int)
    Ind_colname = Ind.columns[-1]



    merged_df = merge_df(y,x,cv,Ind)

    merged_df['interaction'] = merged_df[x_colname] * merged_df[Ind_colname]


    df = merged_df


    #回归
    df = df.dropna()
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


    #分组回归

    # 分组

    ## 低漂绿组（<=平均值）
    low_df = df[df['Pollution'] == 0]
    print("=====" * 10, "Non_Pollution", "=====" * 10)

    formula = 'ESG_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'

    model_stage1 = smf.ols(formula=formula, data=low_df).fit(cov_type='cluster', cov_kwds={'groups': low_df['indcd']})

    print("=====" * 10, "model_stage1", "=====" * 10)
    print(model_stage1.summary())

    formula = 'E_Uncert ~ 1 + ' + x_colname  + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage2 = smf.ols(formula=formula, data=low_df).fit(cov_type='cluster', cov_kwds={'groups': low_df['indcd']})
    print("=====" * 10, "model_stage2", "=====" * 10)
    print(model_stage2.summary())

    formula = 'S_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage3 = smf.ols(formula=formula, data=low_df).fit(cov_type='cluster', cov_kwds={'groups': low_df['indcd']})
    print("=====" * 10, "model_stage3", "=====" * 10)
    print(model_stage3.summary())

    formula = 'G_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage4 = smf.ols(formula=formula, data=low_df).fit(cov_type='cluster', cov_kwds={'groups': low_df['indcd']})
    print("=====" * 10, "model_stage4", "=====" * 10)
    print(model_stage4.summary())


    #高漂绿组
    ## 低漂绿组（<=平均值）
    high_df = df[df['Pollution'] == 1]
    print("=====" * 10, "high_greenwash", "=====" * 10)
    'Assets','Debt'
    formula = 'ESG_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage1 = smf.ols(formula=formula, data=high_df).fit(cov_type='cluster', cov_kwds={'groups': high_df['indcd']})

    print("=====" * 10, "model_stage1", "=====" * 10)
    print(model_stage1.summary())

    formula = 'E_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage2 = smf.ols(formula=formula, data=high_df).fit(cov_type='cluster', cov_kwds={'groups': high_df['indcd']})
    print("=====" * 10, "model_stage2", "=====" * 10)
    print(model_stage2.summary())

    formula = 'S_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage3 = smf.ols(formula=formula, data=high_df).fit(cov_type='cluster', cov_kwds={'groups': high_df['indcd']})
    print("=====" * 10, "model_stage3", "=====" * 10)
    print(model_stage3.summary())

    formula = 'G_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage4 = smf.ols(formula=formula, data=high_df).fit(cov_type='cluster', cov_kwds={'groups': high_df['indcd']})
    print("=====" * 10, "model_stage4", "=====" * 10)
    print(model_stage4.summary())




#X有两个，news_num和meansenti
    #Y有三个，ESG_Uncertainty,ESG_Uncertainy_others,ESG_Uncertainy_others2
    #CV：['Inst_invest','Inde_director','Ten_share','Duality','Analyst',
                            # 'Assets','Debt','ROA','Fixed','Bktomk','Dig_Level','Age','Boardsize']
#之后换x,换y，换cv，就直接调整原始数据和这行命令就行，不用再换了头疼。

first_stage_regression(y='ESG_Uncertainy_others2',x='newsnum',Ind='CIE',cvs=['Boardsize','Age','Inst_invest','Inde_director',
                                        'Ten_share','Duality','Assets','Debt','ROA','Fixed'])



