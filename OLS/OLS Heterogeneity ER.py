import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
#东部地区：北京市、天津市、河北省、山东省、江苏省、上海市、浙江省、福建省、广东省、海南省1
# 还未修改
# 地区与CODE对应：['广东省', '北京市', '吉林省', '江苏省', '辽宁省', '安徽省', '四川省', '湖南省', '浙江省', '河北省',
#        '新疆维吾尔自治区', '山东省', '河南省', '重庆市', '江西省', '山西省', '青海省', '湖北省', '内蒙古自治区',
#        '海南省', '陕西省', '福建省', '广西壮族自治区', '天津市', '云南省', '贵州省', '甘肃省', '宁夏回族自治区',
#        '黑龙江省', '上海市', '西藏自治区', '香港特别行政区', '开曼群岛']
def merge_df(y,x,cv,hv):
    merged_df = pd.merge(y, x, on=['Scode', 'Year'], how='inner')
    merged_df = pd.merge(merged_df, cv, on=['Scode', 'Year'], how='inner')
    merged_df = pd.merge(merged_df,hv,on=['province'], how='inner')
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

    ER = pd.read_csv("../data/Heterogeneity/ER.csv")  # 2015-2020
    ER = ER[['Prov_Code', 'ER']]

    #ER = 1东部, ER = 1 非东部
    ER = ER.dropna()
    ER['Prov_Code'] = ER['Prov_Code'].astype(int)
    # Ind['Year'] = Ind['Year'].astype(int)
    ER_colname = ER.columns[-1]


    ER.columns= ['province','ER']
    merged_df = merge_df(y,x,cv,ER)

    # merged_df['interaction'] = merged_df[x_colname] * merged_df[Ind_colname]#异质性没有interaction


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

    ## 非东部（=0）
    NonE_df = df[df['ER'] == 0]
    print("=====" * 10, "非东部组", "=====" * 10)

    formula = 'ESG_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'

    model_stage1 = smf.ols(formula=formula, data=NonE_df).fit(cov_type='cluster', cov_kwds={'groups': NonE_df['indcd']})

    print("=====" * 10, "model_stage1", "=====" * 10)
    print(model_stage1.summary())

    formula = 'E_Uncert ~ 1 + ' + x_colname  + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage2 = smf.ols(formula=formula, data=NonE_df).fit(cov_type='cluster', cov_kwds={'groups': NonE_df['indcd']})
    print("=====" * 10, "model_stage2", "=====" * 10)
    print(model_stage2.summary())

    formula = 'S_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage3 = smf.ols(formula=formula, data=NonE_df).fit(cov_type='cluster', cov_kwds={'groups': NonE_df['indcd']})
    print("=====" * 10, "model_stage3", "=====" * 10)
    print(model_stage3.summary())

    formula = 'G_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage4 = smf.ols(formula=formula, data=NonE_df).fit(cov_type='cluster', cov_kwds={'groups': NonE_df['indcd']})
    print("=====" * 10, "model_stage4", "=====" * 10)
    print(model_stage4.summary())


    ## 东部组（=1）
    E_df = df[df['ER'] == 1]
    print("=====" * 10, "东部组", "=====" * 10)
    formula = 'ESG_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage1 = smf.ols(formula=formula, data=E_df).fit(cov_type='cluster', cov_kwds={'groups': E_df['indcd']})

    print("=====" * 10, "model_stage1", "=====" * 10)
    print(model_stage1.summary())

    formula = 'E_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage2 = smf.ols(formula=formula, data=E_df).fit(cov_type='cluster', cov_kwds={'groups': E_df['indcd']})
    print("=====" * 10, "model_stage2", "=====" * 10)
    print(model_stage2.summary())

    formula = 'S_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage3 = smf.ols(formula=formula, data=E_df).fit(cov_type='cluster', cov_kwds={'groups': E_df['indcd']})
    print("=====" * 10, "model_stage3", "=====" * 10)
    print(model_stage3.summary())

    formula = 'G_Uncert ~ 1 + ' + x_colname + " + " + " + ".join(cvs) + ' + C(Year) + C(indcd)'
    model_stage4 = smf.ols(formula=formula, data=E_df).fit(cov_type='cluster', cov_kwds={'groups': E_df['indcd']})
    print("=====" * 10, "model_stage4", "=====" * 10)
    print(model_stage4.summary())




#X有两个，news_num和meansenti
    #Y有三个，ESG_Uncertainty,ESG_Uncertainy_others,ESG_Uncertainy_others2
    #CV：['Inst_invest','Inde_director','Ten_share','Duality','Analyst',
                            # 'Assets','Debt','ROA','Fixed','Bktomk','Dig_Level','Age','Boardsize']
#之后换x,换y，换cv，就直接调整原始数据和这行命令就行，不用再换了头疼。

first_stage_regression(y='ESG_Uncertainy_others2',x='newsnum',cvs=['Boardsize','Age','Inst_invest','Inde_director',
                                        'Ten_share','Duality','Assets','Debt','ROA','Fixed'])



