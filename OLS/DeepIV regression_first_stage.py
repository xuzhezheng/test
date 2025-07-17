import keras
from econml.iv.nnet import DeepIV
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
import keras.layers as L

from keras.models import Model
from evaluate import evaluate_prediction



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

    # IV
    # df2和df3适合情感作为x，不然数据量太少，不容易显著
    # 理论上可以迁移一下，那x的过去（3年和行业平均作为IV）
    # df2 = pd.read_csv('../data/IV/企业环保新闻senti_b3.csv')
    # print(df2)
    # df2 = df2[['Scode','Year','e_senti_3yr_avg']]
    # df3 = pd.read_csv('../data/IV/企业环保新闻esenti-同行业同省平均情感值.csv')
    # df3 = df3[['Scode','Year','peer_avg_e_senti']]
    # iv = pd.merge(df2, df3, on=['Scode', 'Year'], how='inner')

    # df4 = pd.read_csv('../data/IV/G_rec.csv')
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
        df5[df5.columns[-1]] = np.log(df5[df5.columns[-1]] + 1)
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
    y.columns = ['Scode', 'Year', 'ESG_Uncert']
    y = y.dropna()
    y_colname = y.columns[-1]

    merged_df = merge_df(y, x, iv, cv)

    # 回归


    df = merged_df[['Scode', 'Year', y_colname, x_colname, 'indcd', 'province'] + iv_list + cvs]
    df = df.dropna()

    X = df[cvs]
    Y = df['ESG_Uncert']
    Z = df[iv_list]
    T = np.log1p(df[x_colname])



    treatment_model = keras.Sequential([keras.layers.Dense(32, activation='relu', input_shape=(13,)),
                                        keras.layers.Dropout(0.1),
                                        keras.layers.Dense(16, activation='relu'),
                                        keras.layers.Dropout(0.1)
                                        ])
    response_model = keras.Sequential([keras.layers.Dense(32, activation='relu', input_shape=(11,)),
                                      keras.layers.Dropout(0.1),
                                       keras.layers.Dense(16, activation='relu'),
                                       keras.layers.Dropout(0.1),
                                      keras.layers.Dense(1)])
    keras_fit_options = {"epochs": 50,
                         "validation_split": 0.0,
                         "callbacks": [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]}
    est = DeepIV(n_components=10, # Number of gaussians in the mixture density networks)
                 m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])), # Treatment model
                 h=lambda t, x: response_model(keras.layers.concatenate([t, x])), # Response model
                 n_samples=1, # Number of samples used to estimate the response
                 first_stage_options = keras_fit_options,
                 second_stage_options = keras_fit_options
                 )

    # boot_est = BootstrapEstimator(wrapped=est, n_bootstrap_samples=5)
    # boot_est.fit(Y, T, X=X, Z=Z)  # Z -> instrumental variables
    # # print(boot_est._instances)
    # T0 = np.zeros((len(X), 1))
    # T1 = np.ones((len(X), 1)) * 10
    # summarize_deepiv_bootstrap(boot_est, X=X, T0=T0, T1=T1)

    est.fit(Y, T, X=X, Z=Z)

    T0 = np.zeros((len(X), 1))
    T1 = np.ones((len(X), 1))
    X = np.array(X)
    Z = np.array(Z)
    X = X.reshape(X.shape[0],-1)
    Z = Z.reshape(Z.shape[0],-1)
    #
    X_in = L.Input((X.shape[1],))
    Z_in = L.Input((Z.shape[1],))




    mog_extractor = Model(inputs=est.model.input,outputs=[est.pi_out, est.mu_out, est.sig_out])
    pi, mu, sig = mog_extractor.predict([Z, X, T])
    print(mu)
    print("pi[0]:", pi[0])
    print("sum(pi[0]):", np.sum(pi[0]))

    print("mu[0]:", mu[0].flatten())  # 展开一个样本
    print("sig[0]:", sig[0])

    T_hat = np.sum(pi[..., np.newaxis] * mu, axis=1)

    r2, rmse, rmae= evaluate_prediction(T,T_hat)
    print(r2,rmse,rmae)


    # Treatment_network()
    # treatment_effects = est.effect(X)
    # print(treatment_effects)
    # inference_result = est.ate_infer(X, T0=T0, T1=T1)
    # print(type(inference_result))
    # print(dir(inference_result))
    # print(inference_result.point_estimate())
    # print(inference_result.pvalue())
    # print(inference_result.stderr())
    # print(inference_result.scale())
    # print(inference_result.zstat())
    # print(inference_result.var())
    # print(inference_result.conf_int())

    # est.model_m.save("deepiv_model_m.h5")

    # # 保存第二阶段 h(t, x) -> y
    # est._h.save("deepiv_model_h.h5")

    # print("effect")
    # print(est.effect(X,T0=T0,T1=T1))#mean point系数
    # print("effect_inference")
    # print(est.effect_inference(X,T0=T0,T1=T1))#全部打印
    # print("effect_interval")
    # print(est.effect_interval(X,T0=T0,T1=T1))#置信区间






    #
    # print("ate")
    # print(est.ate(X,T0=T0,T1=T1))#mean point系数
    # print("ate_interval")
    # print(est.ate_interval(X,T0=T0,T1=T1))#置信区间

    # print("cate_feature_names")
    # print(est.cate_feature_names())#y_name
    # print("cate_output_names")
    # print(est.cate_output_names())#y_name
    # print("cate_treatment_names")
    # print(est.cate_treatment_names())#T_name


DeepIV_regression(x='newsnum',ivs='paper_count',cvs=['Boardsize','Age','Inst_invest','Inde_director',
                                            'Ten_share','Duality','Assets','Debt','ROA','Fixed'])