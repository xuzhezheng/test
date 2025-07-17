import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.api as sm
import seaborn as sns
import numpy as np
def nonlinear_test(merged_df,x,y):
    # 画散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)

    # 添加坐标轴标签
    plt.xlabel(merged_df.columns[-1])
    plt.ylabel(merged_df.columns[-2])

    # 添加标题（可选）
    plt.title('Scatter Plot of Last Two Columns After Merge')

    plt.tight_layout()
    plt.show()
    # 显示合并后的结果
    print(merged_df)
    corr, pval = spearmanr(x, y)
    print(f"Spearman 秩相关系数: {corr:.4f}, p值: {pval:.4e}")
    model = sm.OLS(y, x).fit()
    print(model.summary())
    sns.regplot(x=x, y=y, lowess=True)
    # plt.xlabel("你的X变量名")
    # plt.ylabel("你的Y变量名")
    # plt.title("非线性趋势图")
    plt.show()


# 高管新闻IV
def CEO_news_IV(df1):
    df2 = pd.read_csv('./data/IV/上市公司高管网络新闻_count.csv')
    # 合并：保留 Scode 和 Year 都匹配的行（即“交集”）
    merged_df = pd.merge(df1, df2, on=['Scode', 'Year'], how='inner')
    # x_mean = merged_df.iloc[:, -1].mean()
    # x_std = merged_df.iloc[:, -1].std()
    # x = (merged_df.iloc[:, -1] - x_mean) / x_std

    x = np.log1p(merged_df.iloc[:,-1])
    y = np.log1p(merged_df.iloc[:, -2])
    nonlinear_test(merged_df,x,y)

def G_rec_IV(df1):
    df2 = pd.read_csv('./data/IV/G_rec.csv')
    merged_df = pd.merge(df1, df2, on=['Scode', 'Year'], how='inner')

    x = np.log1p(merged_df.iloc[:, -1])
    y = np.log1p(merged_df.iloc[:, -2])
    nonlinear_test(merged_df, x, y)


def count_b3_IV(df1):
    df2 = pd.read_csv('./data/IV/企业环保新闻count_b3.csv')


    x = np.log1p(df2.iloc[:, -1])
    y = np.log1p(df2.iloc[:, -2])
    nonlinear_test(df2, x, y)

def prov_ind_IV(df1):
    df2 = pd.read_csv('./data/IV/企业环保新闻count-同行业同省平均值.csv')


    x = np.log1p(df2.iloc[:, -1])
    y = np.log1p(df2.iloc[:, -2])
    nonlinear_test(df2, x, y)



df1 = pd.read_csv('./data/X/news_num.csv')
# CEO_news_IV(df1)
# G_rec_IV(df1)
# count_b3_IV(df1)
prov_ind_IV(df1)


