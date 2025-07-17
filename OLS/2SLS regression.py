import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from scipy.stats import entropy
from scipy.stats.mstats import winsorize
from statsmodels.stats.outliers_influence import variance_inflation_factor

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



def merge_df(y,x,iv,cv):
    merged_df = pd.merge(y, x, on=['Scode', 'Year'], how='inner')


    merged_df = pd.merge(merged_df, cv, on=['Scode', 'Year'], how='inner')
    print(iv)
    merged_df = pd.merge(merged_df, iv, on=['Scode', 'Year'], how='inner')
    print(merged_df)
    return merged_df

def compute_entropy(row):
    counts = np.array([row['pos_news_cnt'], row['neu_news_cnt'], row['neg_news_cnt']])
    total = counts.sum()
    if total == 0:
        return np.nan  # 没有新闻，不计算熵
    probs = counts / total
    return entropy(probs, base=2)  # 使用 log2 为单位

def na_list(df):
    null_columns = df.columns[df.isnull().any()]
    print(null_columns)


def two_stage_regression(y='ESG_Uncertainy_others2',x='senti',ivs = 'network_mean' ,cvs=['Inst_invest','Inde_director','Ten_share','Duality','Analyst','Assets','Debt','ROA','Fixed','Bktomk']):
    # X有两个，news_num和meansenti
    # Y有三个，ESG_Uncertainty,ESG_Uncertainy_others,ESG_Uncertainy_others2
    # CV：['Inst_invest','Inde_director','Ten_share','Duality',
    #                 'Analyst','Assets','Debt','ROA','Fixed','Bktomk']
    #IV有三个先全部都作为IV，不用判断
    #X
    if x == 'senti':
        x = pd.read_csv('../data/X/企业环保新闻senti.csv')
    elif x =='newsnum':
        x = pd.read_csv("../data/X/news_num.csv")
        x['Scode'] = x['Scode'].astype(int)
        x['Year'] = x['Year'].astype(int)
    else:
        #这个是Datago处理后的每年新闻数，正面新闻数等
        x = pd.read_csv("./x/cnt_poscnt_negcnt_maxsenti_minsenti_meansenti_stdsenti.csv")
    x_colname = x.columns[-1]

    #IV
    #df2和df3适合情感作为x，不然数据量太少，不容易显著
    #理论上可以迁移一下，那x的过去（3年和行业平均作为IV）
    # df2 = pd.read_csv('../data/IV/企业环保新闻senti_b3.csv')
    # print(df2)
    # df2 = df2[['Scode','Year','e_senti_3yr_avg']]
    # df3 = pd.read_csv('../data/IV/企业环保新闻esenti-同行业同省平均情感值.csv')
    # df3 = df3[['Scode','Year','peer_avg_e_senti']]
    # iv = pd.merge(df2, df3, on=['Scode', 'Year'], how='inner')


    df4 = pd.read_csv('../data/IV/G_rec.csv')
    df6 = pd.read_csv('../data/IV/企业环保新闻count_b3.csv')
    df6 = df6[['Scode','Year','newsnum_3yr_avg']]
    df6['newsnum_3yr_avg'] = np.log(df6['newsnum_3yr_avg'] + 1)

    df7 = pd.read_csv('../data/IV/企业环保新闻count-同行业同省平均值.csv')
    df7 = df7[['Scode', 'Year', 'peer_avg_news_count']]
    df7['peer_avg_news_count'] = np.log(df7['peer_avg_news_count'] + 1)

    iv = pd.merge(df4, df6, on=['Scode', 'Year'], how='inner')
    iv = pd.merge(iv, df7, on=['Scode', 'Year'], how='inner')
    # iv_list = ['G_rec','newsnum_3yr_avg','peer_avg_news_count']
    iv_list = ['newsnum_3yr_avg','peer_avg_news_count']
    # iv_list = ['newsnum_3yr_avg']
    # iv_list = ['peer_avg_news_count']
    # iv_list = []
    if ivs == 'network_mean':
        df5 = pd.read_csv('../data/IV/上市公司高管网络新闻_mean.csv')
        df5.columns = ['Scode', 'Year', 'G_news']
        iv = pd.merge(iv, df5, on=['Scode', 'Year'], how='inner')
        iv_list.append(df5.columns[-1])
    elif ivs == 'network_count':
        df5 = pd.read_csv('../data/IV/上市公司高管网络新闻_count.csv')
        # df5[df5.columns[-1]] = np.log(df5[df5.columns[-1]])
        df5.columns = ['Scode', 'Year', 'G_news']
        iv = pd.merge(iv, df5, on=['Scode', 'Year'], how='inner')
        iv_list.append(df5.columns[-1])
    elif ivs == 'paper_mean':
        df5 = pd.read_csv('../data/IV/上市公司高管报刊新闻_mean.csv')
        df5.columns = ['Scode', 'Year', 'G_news']
        iv = pd.merge(iv, df5, on=['Scode', 'Year'], how='inner')
        iv_list.append(df5.columns[-1])
    elif ivs == 'paper_count':
        df5 = pd.read_csv('../data/IV/上市公司高管报刊新闻_count.csv')
        df5[df5.columns[-1]] = np.log(df5[df5.columns[-1]]+1)
        df5.columns = ['Scode', 'Year', 'G_news']
        df5['G_news'] = winsorize(df5['G_news'], limits=[0.01, 0.01])
        iv = pd.merge(iv, df5, on=['Scode', 'Year'], how='inner')
        iv_list.append(df5.columns[-1])

    #control variable
    # cv = pd.read_csv("../data/CV/control_variable.csv")
    # cv = pd.read_csv("../data/CV/control_variable2.csv")
    cv = pd.read_csv("../data/CV/control_variable3.csv")


    #y
    if y == 'ESG_Uncertainty':
        y = pd.read_csv("../data/Y/ESG_Uncertainty.csv")
    elif y == 'ESG_Uncertainy_others':
        y = pd.read_csv("../data/Y/ESG_Uncertainty_others.csv")
    elif y == 'ESG_Uncertainy_others2':
        y = pd.read_csv("../data/Y/ESG_Uncertain_others2.csv")
    y.columns = ['Scode', 'Year', 'ESG_Uncert', 'E_Uncert', 'S_Uncert', 'G_Uncert']
    y = y.dropna()
    y_colname = y.columns[-4]


    merged_df = merge_df(y,x,iv,cv)
    # df = merged_df[['Scode', 'Year', x_colname, 'ESG_Uncert',
    #                 'e_senti_3yr_avg','peer_avg_e_senti','G_rec'] + cvs]


    #回归
    na_list(merged_df)

    df = merged_df[['Scode', 'Year', y_colname,x_colname,'indcd','province'] + iv_list + cvs]
    cvs = drop_high_vif(df, cvs, threshold=10)
    df = df.dropna()
    print(df)
    # print(df)
    X = df[x_colname]
    # X = sm.add_constant(X)  # 控制变量

    # IV = df[['e_senti_3yr_avg', 'peer_avg_e_senti', 'G_rec']]
    # # IV = df[['e_senti_3yr_avg', 'peer_avg_e_senti']]
    # IV = sm.add_constant(IV)  # 工具变量
    # Y = df['ESG_Uncert']
    #
    # CV = df[cvs]
    #不加固定效应
    # formula = 'ESG_Uncert ~ ' + " + ".join(cvs) + " [" + x_colname + " ~ e_senti_3yr_avg + peer_avg_e_senti + G_news] "
    # model = IV2SLS.from_formula(formula,data=df).fit(cov_type='clustered', clusters=df['Year'])

    #加聚类标准误
    print(iv_list)
    if ivs !='':
        # formula = 'ESG_Uncert ~ ' + " + ".join(cvs) + " [" + x_colname + " ~ " + " + ".join(iv_list) + " + G_news]"
        formula = 'ESG_Uncert ~ 1 + ' + " + ".join(cvs) + " [" + x_colname + " ~ " + " + ".join(iv_list) + "] + C(Year) + C(indcd)"
    else:
        formula = 'ESG_Uncert ~ 1 + ' + " + ".join(cvs) + " [" + x_colname + " ~ " + " + ".join(iv_list) + "] + C(Year) + C(indcd)"
        # formula = 'ESG_Uncert ~ ' + " + ".join(cvs) + " [" + x_colname + " ~ " + " + ".join(
        #     iv_list) + "] + C(Year) + C(indcd)"
    #稳健标准误
    # model = IV2SLS.from_formula(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['indcd']})
    #聚类标准误

    model = IV2SLS.from_formula(formula,data=df).fit(cov_type='clustered', clusters=df['indcd'])
    # model = IVGMM.from_formula(formula, data=df).fit(cov_type='clustered', clusters=df['indcd'])
    print(model.summary)
    print(model.first_stage.summary)
     # 'overidentification'



#
two_stage_regression(x='newsnum',ivs='paper_mean',cvs=['Boardsize','Age','Inst_invest','Inde_director',
                                        'Ten_share','Duality','Assets','Debt','ROA','Fixed'])




