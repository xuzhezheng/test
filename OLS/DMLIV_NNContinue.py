from sklearn.neural_network import MLPRegressor,MLPClassifier
import keras
from econml.iv.dml import DMLIV
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
import keras.layers as L
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Model
from evaluate import evaluate_prediction
from sklearn.linear_model import LinearRegression,Ridge
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def predict_t_expectation(pi, mu):
    """
    根据混合高斯分布参数计算期望 T 值

    参数：
        pi: ndarray, shape (batch_size, n_components)
        mu: ndarray, shape (batch_size, n_components, d_t)

    返回：
        t_hat: ndarray, shape (batch_size, d_t)
    """
    # 添加新维度使 pi 可与 mu 广播相乘：pi -> (batch_size, n_components, 1)
    weighted_mu = pi[..., np.newaxis] * mu  # shape: (batch_size, n_components, d_t)

    # 沿 n_components 维度求和，得到最终的 t_hat
    t_hat = np.sum(weighted_mu, axis=1)  # shape: (batch_size, d_t)

    return t_hat
def merge_df(y,x,iv,cv):
    merged_df = pd.merge(y, x, on=['Scode', 'Year'], how='inner')
    print("mergeddf1:",merged_df)
    merged_df = pd.merge(merged_df, cv, on=['Scode', 'Year'], how='inner')
    print("mergeddf2:", merged_df)
    merged_df = pd.merge(merged_df, iv, on=['Scode', 'Year'], how='inner')
    print("mergeddf3:", merged_df)
    return merged_df


def DeepIV_regression(y='ESG_Uncertainy_others2',x='senti',ivs = 'network_mean' ,cvs=['Inst_invest','Inde_director','Ten_share','Duality','Analyst','Assets','Debt','ROA','Fixed','Bktomk']):
    # X有两个，news_num和meansenti
    # Y有三个，ESG_Uncertainty,ESG_Uncertainy_others,ESG_Uncertainy_others2
    # CV：['Inst_invest','Inde_director','Ten_share','Duality',
    #                 'Analyst','Assets','Debt','ROA','Fixed','Bktomk']
    # IV有三个先全部都作为IV，不用判断
    # X
    if x == 'senti':
        x = pd.read_csv('../data/X/企业环保新闻senti.csv')
    elif x == 'newsnum':
        x = pd.read_csv("../data/X/news_num.csv")
        x['Scode'] = x['Scode'].astype(int)
        x['Year'] = x['Year'].astype(int)
    else:
        # 这个是Datago处理后的每年新闻数，正面新闻数等
        x = pd.read_csv("./x/cnt_poscnt_negcnt_maxsenti_minsenti_meansenti_stdsenti.csv")
    x_colname = x.columns[-1]

    df6 = pd.read_csv('../data/IV/企业环保新闻count_b3.csv')
    df6 = df6[['Scode', 'Year', 'newsnum_3yr_avg']]
    df6['newsnum_3yr_avg'] = np.log1p(df6['newsnum_3yr_avg'])

    df7 = pd.read_csv('../data/IV/企业环保新闻count-同行业同省平均值.csv')
    df7 = df7[['Scode', 'Year', 'peer_avg_news_count']]
    df7['peer_avg_news_count'] = np.log1p(df7['peer_avg_news_count'])

    # iv = pd.merge(df4, df6, on=['Scode', 'Year'], how='inner')
    iv = pd.merge(df6, df7, on=['Scode', 'Year'], how='inner')
    print("iv:",iv)
    # iv_list = ['G_rec','newsnum_3yr_avg','peer_avg_news_count']
    iv_list = ['newsnum_3yr_avg', 'peer_avg_news_count']
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
        df5[df5.columns[-1]] = np.log1p(df5[df5.columns[-1]])
        df5.columns = ['Scode', 'Year', 'G_news']
        df5['G_news'] = winsorize(df5['G_news'], limits=[0.01, 0.01])
        iv = pd.merge(iv, df5, on=['Scode', 'Year'], how='inner')
        iv_list.append(df5.columns[-1])
    print("iv2:",iv)
    # control variable
    # cv = pd.read_csv("../data/CV/control_variable.csv")
    # cv = pd.read_csv("../data/CV/control_variable2.csv")
    cv = pd.read_csv("../data/CV/control_variable3.csv")

    # y
    if y == 'ESG_Uncertainty':
        y = pd.read_csv("../data/Y/ESG_Uncertainty.csv")
    elif y == 'ESG_Uncertainy_others':
        y = pd.read_csv("../data/Y/ESG_Uncertainty_others.csv")
    elif y == 'ESG_Uncertainy_others2':
        y = pd.read_csv("../data/Y/ESG_Uncertain_others2.csv")
    y.columns = ['Scode', 'Year', 'ESG_Uncert', 'E_Uncert', 'S_Uncert', 'G_Uncert']
    y = y[['Scode','Year','ESG_Uncert']]
    y = y.dropna()
    y_colname = y.columns[-1]

    merged_df = merge_df(y, x, iv, cv)

    # 回归


    df = merged_df[['Scode', 'Year', y_colname, x_colname, 'indcd', 'province'] + iv_list + cvs]
    df = df.dropna()
    df = df[df['indcd'] != 17]

    df = pd.get_dummies(df,columns=['indcd', 'Year'], drop_first=True)
    print(df.columns)


    X = df[cvs+['indcd_2',
       'indcd_3', 'indcd_4', 'indcd_5', 'indcd_6', 'indcd_7', 'indcd_8',
       'indcd_9', 'indcd_10', 'indcd_11', 'indcd_12', 'indcd_13', 'indcd_14',
       'indcd_15', 'indcd_16', 'indcd_18', 'Year_2011',
       'Year_2012', 'Year_2013', 'Year_2014', 'Year_2015', 'Year_2016',
       'Year_2017', 'Year_2018', 'Year_2019', 'Year_2020', 'Year_2021',
       'Year_2022']]
    Y = df['ESG_Uncert']
    Z = df[iv_list]
    T = winsorize(df[x_colname]-1, limits=[0.01, 0.01])


    est = DMLIV(
    model_y_xw=RandomForestRegressor(random_state=42),
        model_t_xwz=MLPRegressor(hidden_layer_sizes=(64,32),
                      activation='relu',
                      solver='adam',
                      max_iter=500,
                      random_state=42),
        model_t_xw=MLPRegressor(hidden_layer_sizes=(64,32),
                      activation='relu',
                      solver='adam',
                      max_iter=500,
                      random_state=42),
        model_final=LinearRegression(),
    discrete_treatment=False,
    cv=10)


    est.fit(Y=Y, T=T, X=X, Z=Z,inference='bootstrap')





    T_pred = est.models_t_xwz[0][0].predict(np.concatenate([X,Z], axis=1))
    plt.plot(T)
    plt.plot(T_pred)
    plt.xlabel("Sample index")
    plt.ylabel("T value")
    plt.legend()
    plt.show()
    r2 = r2_score(T, T_pred)
    rmse = mean_squared_error(T, T_pred, squared=False)
    print("r2:", r2)

    Y_pred = est.models_y_xw[0][0].predict(X)
    plt.plot(Y)
    plt.plot(Y_pred)
    plt.xlabel("Sample index")
    plt.ylabel("Y value")
    plt.legend()
    plt.show()
    r2 = r2_score(Y, Y_pred)
    print("Y‘s r2:", r2)



    score = est.score(Y=Y, T=T, X=X, Z=Z)
    var = np.var(Y, ddof=0)  # 总体方差
    r2 = 1 - score / var
    print("score of CATE model:", score)
    print("R² of CATE model:", r2)

    with open("../results/DMLNNC_effect_inference_result_x更正.txt", "a", encoding="utf-8") as f:
        f.write("**" * 40)
        f.write("\n")

        effect = est.const_marginal_ate_inference(X)
        f.write(str(effect))
        f.write(str(est.summary()))
        f.write("\n")








# for i in range(50):
#
DeepIV_regression(x='newsnum',ivs='paper_count',cvs=['Boardsize','Age','Inst_invest','Inde_director',
                                            'Ten_share','Duality','Assets','Debt','ROA','Fixed'])
# cvs = ['Boardsize','Age','Inst_invest','Inde_director',
#                                             'Ten_share','Duality','Assets','Debt','ROA','Fixed']
# for i in range(1,len(cvs)):
#     DeepIV_regression(x='newsnum',ivs='paper_count',cvs=cvs[:i])