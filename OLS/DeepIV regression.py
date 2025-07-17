import joblib
import keras
from econml.iv.nnet import DeepIV
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from econml.inference._inference import BootstrapEstimator

from scipy.stats import norm

def summarize_deepiv_bootstrap(boot_est, X, T0, T1):
    ates = []
    for model in boot_est._instances:
        eff = model.effect(X=X, T0=T0, T1=T1)
        ates.append(eff.mean())

    ates = np.array(ates)
    mean = ates.mean()
    stderr = ates.std(ddof=1)
    zstat = mean / stderr
    pval = 2 * (1 - norm.cdf(abs(zstat)))
    ci = np.percentile(ates, [2.5, 97.5])

    # 星号表示
    if pval <= 0.01:
        stars = "***"
    elif pval <= 0.05:
        stars = "**"
    elif pval <= 0.1:
        stars = "*"
    else:
        stars = ""

    print("\nBootstrap ATE 推断结果")
    print("="*40)
    print(f"ATE Estimate   : {mean:.4f} {stars}")
    print(f"Std. Error     : {stderr:.4f}")
    print(f"Z-statistic    : {zstat:.2f}")
    print(f"P-value        : {pval:.4f}")
    print(f"95% CI         : [{ci[0]:.4f}, {ci[1]:.4f}]")
    print("="*40)
def merge_df(y,x,iv,cv):
    merged_df = pd.merge(y, x, on=['Scode', 'Year'], how='inner')
    merged_df = pd.merge(merged_df, cv, on=['Scode', 'Year'], how='inner')
    merged_df = pd.merge(merged_df, iv, on=['Scode', 'Year'], how='inner')
    return merged_df

def treatment_model(z,x):
    input_layer = keras.layers.Concatenate()([z, x])
    hidden = keras.layers.Dense(64, activation='relu')(input_layer)
    hidden = keras.layers.Dropout(0.17)(hidden)
    hidden = keras.layers.Dense(32, activation='relu')(hidden)
    output = keras.layers.Dropout(0.17)(hidden)
    return keras.Model(inputs=[z,x], outputs=output)

def response_model(t,x):
    input_layer = keras.layers.Concatenate()([t, x])
    hidden = keras.layers.Dense(64, activation='relu')(input_layer)
    hidden = keras.layers.Dropout(0.17)(hidden)
    hidden = keras.layers.Dense(32, activation='relu')(hidden)
    hidden = keras.layers.Dropout(0.17)(hidden)
    output =  keras.layers.Dense(1)(hidden)
    return keras.Model(inputs=[t, x], outputs=output)

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

    df4 = pd.read_csv('../data/IV/G_rec.csv')
    df6 = pd.read_csv('../data/IV/企业环保新闻count_b3.csv')
    df6 = df6[['Scode', 'Year', 'newsnum_3yr_avg']]
    df6['newsnum_3yr_avg'] = np.log(df6['newsnum_3yr_avg'] + 1)

    df7 = pd.read_csv('../data/IV/企业环保新闻count-同行业同省平均值.csv')
    df7 = df7[['Scode', 'Year', 'peer_avg_news_count']]
    df7['peer_avg_news_count'] = np.log(df7['peer_avg_news_count'] + 1)

    iv = pd.merge(df4, df6, on=['Scode', 'Year'], how='inner')
    iv = pd.merge(iv, df7, on=['Scode', 'Year'], how='inner')
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

    # control variable
    # cv = pd.read_csv("../data/CV/control_variable.csv")
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
    # print(df)
    X = df[cvs]
    Y = df['ESG_Uncert']
    Z = df[iv_list]
    T = df[x_colname]



    treatment_model = keras.Sequential([keras.layers.Dense(32, activation='relu', input_shape=(len(iv_list)+10,)),
                                        keras.layers.Dropout(0.17),
                                        keras.layers.Dense(16, activation='relu'),
                                        keras.layers.Dropout(0.17)])
    response_model = keras.Sequential([keras.layers.Dense(32, activation='relu', input_shape=(11,)),
                                      keras.layers.Dropout(0.17),
                                      keras.layers.Dense(16, activation='relu'),
                                      keras.layers.Dropout(0.17),
                                      keras.layers.Dense(1)])
    keras_fit_options = {"epochs": 10,
                         "validation_split": 0.1,
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

    est.fit(Y, T, X=X, Z=Z, inference='bootstrap')

    T0 = np.zeros((len(X), 1))
    T1 = np.ones((len(X), 1))
    inference_result = est.ate_inference(X, T0=T0, T1=T1)
    # print(type(inference_result))
    # print(dir(inference_result))
    # print(inference_result.point_estimate())
    # print(inference_result.pvalue())
    # print(inference_result.stderr())
    # print(inference_result.scale())
    # print(inference_result.zstat())
    # print(inference_result.var())
    # print(inference_result.conf_int())

    with open("effect_inference_result_x更正.txt", "a", encoding="utf-8") as f:
        f.write(str(inference_result))
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

for i in range(50):
    DeepIV_regression(x='newsnum',ivs='paper_count',cvs=['Boardsize','Age','Inst_invest','Inde_director',
                                            'Ten_share','Duality','Assets','Debt','ROA','Fixed'])