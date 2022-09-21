import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xlwt
import tensorflow as tf
from PIL._imaging import display
from numpy import mean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from pandas._testing import assert_frame_equal
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

# 不加这个无法绘图
plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
plt.rcParams['font.size'] = 12  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

#读取xls文件
def load_data():  # 读取文件

    df = pd.read_excel(r'..\data\期货价格（多变量）.xls', index_col='日期')

    return df

#写入xls文件
def write_data(data):  # 写入文件

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    x = pd.DataFrame(data)
    x.to_excel(r'F:\QH\Data\College\My Race\SRT\futures_prediction\data\汇总（多变量）.xls', sheet_name='多变量')
    savepath = r'F:\QH\Data\College\My Race\SRT\futures_prediction\data\汇总（多变量）.xls'
    book.save(savepath)

#填补期货价格
def fill_price(df):
    # 多变量填补法
    II=IterativeImputer(max_iter=10,random_state=0)
    df=II.fit_transform(df)
    # KNN填补法
    # KI=KNNImputer(n_neighbors=10,weights="uniform")
    # price=KI.fit_transform(df)

    # 计算每月日均值
    # for k in range(1, 32):  # 每月31天
    #     df = df.replace(np.nan, 0)
    #     # 初始化列表,用来存储一个月每天的价格均值
    #     day1 = [0 for j in range(28)]
    #     i = 0  # i表示列数
    #     for index, col in df.iteritems():
    #         num = 0
    #         count = 0
    #         for index, row in df.iterrows():
    #             if row['DAY'] == k:
    #                 if row[i] != 0:  # 有初始值的项
    #                     num += row[i]
    #                     count += 1
    #         day1[i - 1] = num / count
    #         i += 1
    #     print(day1)
    #     df = df.replace(0, np.nan)
    #     # 填补
    #     i = 1  # 列标
    #     for index, col in df.iteritems():
    #         j = 1  # 行标
    #         for index, row in df.iterrows():
    #             if row['DAY'] == k:
    #                 if i >= 29:  # 退出循环，防止数组越界
    #                     break
    #                 if j >= 3172:
    #                     break
    #                 if pd.isnull(row[i]):
    #                     df.iat[j - 1, i] = day1[i - 1]  # 将求出的均值填充进df
    #             j += 1
    #         i += 1
    #     print('第%d天完成' % (k))
    return df

#填补开工率
def fill_rate(df):
    # df = df.fillna(method='pad')
    # 计算每月日均值
    for k in range(1, 32):  # 每月31天
        df = df.replace(np.nan, 0)
        # 初始化列表,用来存储一个月每天的价格均值
        day1 = [0 for j in range(36)]
        i = 0  # i表示列数
        for index, col in df.iteritems():
            num = 0
            count = 0
            for index, row in df.iterrows():
                if row['DAY'] == k:
                    if row[i] != 0:  # 有初始值的项
                        num += row[i]
                        count += 1
            day1[i - 1] = num / count
            i += 1
        print(day1)
        df = df.replace(0, np.nan)
        # 填补
        i = 1  # 列标
        for index, col in df.iteritems():
            j = 1  # 行标
            for index, row in df.iterrows():
                if row['DAY'] == k:
                    if i >= 37:  # 退出循环，防止数组越界
                        break
                    # if j>=3172:
                    #     break
                    if pd.isnull(row[i]):
                        df.iat[j - 1, i] = day1[i - 1]  # 将求出的均值填充进df
                j += 1
            i += 1
        print('第%d天完成' % (k))
    return df

#Z-score标准化
def stardand(df):
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    return df

#PCA降维与重构
def pca(dataMat, topNfeat=27):  # m=3171,n=64,p=23
    meanVals = np.mean(dataMat, axis=0)  # 均值：1*n
    meanRemoved = dataMat - meanVals  # 去除均值  m*n
    covMat = np.cov(meanRemoved, rowvar=0)  # 协方差矩阵 n*n
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 特征值 1*n，特征矩阵  n*n
    # for i in range(63):#每个柱子是前面所有柱子的累加
    #     eigVals[i+1]+=eigVals[i]
    # sum=eigVals[63]
    # for i in range(64):#求百分比
    #     eigVals[i]/=sum
    # index = np.arange(len(eigVals));
    # plt.bar(index, eigVals, width=0.6)
    # plt.xticks([x for x in range(64) if x % 2 == 0])  # x标记step设置为2
    # for a, b in zip(index, eigVals):  # 柱子上的数字显示
    #     plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=4);
    # plt.show()
    eigValInd = np.argsort(eigVals)  # 将特征值从小到大索引排序
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 从后往前仅保留p个列（将topNfeat理解为p即可）
    redEigVects = pd.DataFrame(eigVects[:, eigValInd])  # 仅保留p个最大特征值对应的特征向量，按从大到小的顺序重组特征矩阵n*p
    lowDDataMat = np.dot(meanRemoved, redEigVects)  # 将数据转换到低维空间lowDDataMat： m*p
    reconMat = np.dot(pd.DataFrame(lowDDataMat), redEigVects.T) + meanRemoved  # 从压缩空间重构原数据reconMat：  m*n
    return lowDDataMat, reconMat


if __name__ == '__main__':
    prall = load_data()
    # pd.set_option('display.max_rows', None)  # 可以填数字，填None表示'行'无限制
    # pd.set_option('display.max_columns', None)  # 可以填数字，填None表示'列'无限制
    df=stardand(prall)
    write_data(df)
    # write_data(df)
    # #PTA与原油
    # plt.figure(figsize=(8,4))
    # sns.lineplot(x='现货价:原油:英国布伦特Dtd',y='期货收盘价(活跃合约):精对苯二甲酸(PTA)',data=df)
    # plt.show()
    # #乙二醇与原油
    # sns.lineplot(x='现货价:原油:英国布伦特Dtd', y='期货收盘价(活跃合约):乙二醇', data=df)
    # plt.show()
    # #短纤与原油
    # sns.lineplot(x='现货价:原油:英国布伦特Dtd',y='期货收盘价(活跃合约):短纤',data=df)
    # plt.show()


