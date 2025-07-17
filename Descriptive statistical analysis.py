import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def merge_df(y,x,cv):
    merged_df = pd.merge(y, x, on=['Scode', 'Year'], how='inner')
    merged_df = pd.merge(merged_df, cv, on=['Scode', 'Year'], how='inner')
    return merged_df



def first_stage_regression(y='ESG_Uncertainy_others2',x='senti',cvs=['Inst_invest','Inde_director','Ten_share','Duality','Analyst','Assets','Debt','ROA','Fixed','Bktomk']):
    #X有两个，news_num和meansenti
    #Y有三个，ESG_Uncertainty,ESG_Uncertainy_others,ESG_Uncertainy_others2
    #CV：['Inst_invest','Inde_director','Ten_share','Duality',
    #                 'Analyst','Assets','Debt','ROA','Fixed','Bktomk']
    if x == 'senti':
        x = pd.read_csv('./data/X/企业环保新闻senti.csv')
    elif x =='newsnum':
        x = pd.read_csv("./data/X/news_num.csv")
        x['Scode'] = x['Scode'].astype(int)
        x['Year'] = x['Year'].astype(int)
    x_colname = x.columns[-1]
    x['news_count'] = np.log(x['news_count'] + 1)

    #cv
    cv = pd.read_csv("./data/CV/control_variable3.csv")

    #y
    if y == 'ESG_Uncertainty':
        y = pd.read_csv("./data/Y/ESG_Uncertainty.csv")
    elif y == 'ESG_Uncertainy_others':
        y = pd.read_csv("./data/Y/ESG_Uncertainty_others.csv")
    elif y == 'ESG_Uncertainy_others2':
        y = pd.read_csv("./data/Y/ESG_Uncertain_others2.csv")
    y.columns = ['Scode','Year','ESG_Uncert']

    merged_df = merge_df(y,x,cv)

    df = merged_df


    #回归
    df = df.dropna()
    summary_stats = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'median': df.median()
    })

    print(summary_stats)
    print(df.shape[0])

    # formula = 'ESG_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    # model_stage1 = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['indcd']})
    #
    #
    # print(model_stage1.summary())


#X有两个，news_num和meansenti
    #Y有三个，ESG_Uncertainty,ESG_Uncertainy_others,ESG_Uncertainy_others2
    #CV：['Inst_invest','Inde_director','Ten_share','Duality','Analyst',
                            # 'Assets','Debt','ROA','Fixed','Bktomk','Dig_Level','Age','Boardsize']
#之后换x,换y，换cv，就直接调整原始数据和这行命令就行，不用再换了头疼。

first_stage_regression(x='newsnum',cvs=['Boardsize','Age','Inst_invest','Inde_director',
                                        'Ten_share','Duality','Assets','Debt','ROA','Fixed'])



