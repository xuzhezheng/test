from sklearn.neural_network import MLPRegressor
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
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import matplotlib.pyplot as plt
from econml.dml import DML,LinearDML
from sklearn.preprocessing import QuantileTransformer


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
def merge_df(y,x,cv):
    merged_df = pd.merge(y, x, on=['Scode', 'Year'], how='inner')
    print("mergeddf1:",merged_df)
    merged_df = pd.merge(merged_df, cv, on=['Scode', 'Year'], how='inner')
    print("mergeddf2:", merged_df)
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


    # control variable
    # cv = pd.read_csv("../data/CV/control_variable.csv")
    # cv = pd.read_csv("../data/CV/control_variable2.csv")
    cv = pd.read_csv("../data/CV/control_variable3.csv")

    # y
    if y == 'ESG_Uncertainy':
        y = pd.read_csv("../data/Y/ESG_Uncertainty.csv")
    elif y == 'ESG_Uncertainy_others':
        y = pd.read_csv("../data/Y/ESG_Uncertainty_others.csv")
    elif y == 'ESG_Uncertainy_others2':
        y = pd.read_csv("../data/Y/ESG_Uncertain_others2.csv")
    y.columns = ['Scode', 'Year', 'ESG_Uncert', 'E_Uncert', 'S_Uncert', 'G_Uncert']
    y = y[['Scode', 'Year', 'ESG_Uncert']]
    y = y.dropna()
    y_colname = y.columns[-1]

    merged_df = merge_df(y, x, cv)

    # 回归


    df = merged_df[['Scode', 'Year', y_colname, x_colname, 'indcd', 'province'] + cvs]
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
    # X = df[cvs+['indcd_2','indcd_3', 'indcd_4', 'indcd_5', 'indcd_6', 'indcd_7', 'indcd_8',
    #    'indcd_9', 'indcd_10', 'indcd_11', 'indcd_12', 'indcd_13', 'indcd_14',
    #    'indcd_15', 'indcd_16', 'indcd_18', 'Year_2016',
    #    'Year_2017', 'Year_2018', 'Year_2019', 'Year_2020', 'Year_2021',
    #    'Year_2022']]
    # X = df[cvs + ['indcd_2', 'indcd_3', 'indcd_4', 'indcd_5', 'indcd_6', 'indcd_7', 'indcd_8',
    #    'indcd_9', 'indcd_11', 'indcd_12', 'indcd_13',
    #    'indcd_15',  'Year_2016',
    #    'Year_2017', 'Year_2018', 'Year_2019', 'Year_2020', 'Year_2021',
    #    'Year_2022']]

    # X = df[cvs]
    # df_numeric = df.select_dtypes(include='number')
    # df[df_numeric.columns] = (df_numeric - df_numeric.mean()) / df_numeric.std()

    Y = df['ESG_Uncert']
    # T = df[x_colname]/np.max(df[x_colname])
    T = winsorize(df[x_colname], limits=[0.01, 0.01])
    # T = QuantileTransformer().fit_transform(np.array(T).reshape(-1, 1)).flatten()
    # T = pd.Series(T,name=x_colname)
    # param = {
    #     "objective": "regression",
    #     "metric": "rmse",
    #     "verbosity": -1,
    #     "boosting_type": "gbdt",
    #     "learning_rate": 0.15020328298528612,
    #     "num_leaves": 112,
    #     "max_depth": 12,
    #     "min_child_samples": 95,
    #     "subsample": 0.9827488439609978,
    #     "colsample_bytree": 0.7364170727167143,
    #     "reg_alpha": 1.603969153040404,
    #     "reg_lambda": 0.00010460375357666728
    # }
    est = LinearDML(
        model_y=RandomForestRegressor(random_state=42),
        model_t=RandomForestClassifier(random_state=42),
        discrete_treatment= True,
        cv=9
    )
    # plt.figure(figsize=(8, 6))
    # plt.plot(Y, T)
    # plt.show()


    est.fit(Y=Y, T=T, X=X, inference='statsmodels')

    with open("../results/DMLReg_effect_inference_result_x更正.txt", "a", encoding="utf-8") as f:
        print("**"*40)
        for i in range(2, 20):
            effect = est.ate_inference(X, T0=1, T1=i)
            f.write(str(effect))
        print("")

    print(est.summary())
    T_pred = est.models_t[0][0].predict(X)
    r2 = r2_score(T,T_pred)
    print("T R2:",r2)

    Y_pred =est.models_y[0][0].predict(X)
    r2 = r2_score(Y, Y_pred)
    print("Y R2:", r2)
    # print(est.coef_)
    # print(est.cate_feature_names())
    # print(est.nuisance_scores_y_xw)
    # print(est.nuisance_scores_t_xwz)
    # print(est.model_final_.)

    score = est.score(Y=Y, T=T, X=X)
    var = np.var(Y, ddof=0)  # 总体方差
    r2 = 1 - score / var
    print("R² of CATE model:", r2)




# cvs=['Boardsize','Age','Inst_invest','Inde_director',
#                                             'Ten_share','Duality','Assets','Debt','ROA','Fixed']

DeepIV_regression(y='ESG_Uncertainy_others2',x='newsnum',ivs='paper_count',cvs=['Boardsize','Age','Inst_invest','Inde_director',
                                            'Ten_share','Duality','ROA','Fixed'])

# for i in range(1,len(cvs)):
#     DeepIV_regression(x='newsnum',ivs='paper_count',cvs=cvs[:i])