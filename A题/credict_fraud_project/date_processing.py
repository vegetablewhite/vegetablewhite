import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# from chart_studio import plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# import cufflinks
# cufflinks.go_offline(connected=True)
# init_notebook_mode(connected=True)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.gridspec as gridspec


from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import ADASYN


def date_processing(c_path):
    date = pd.read_csv(c_path, engine='python')

    # print(date['distance_from_home'].max())
    # print(date['distance_from_home'].min())

    # 可以看到最大值与最小值差值很大，需要对前三列进行标准化处理
    #标准化过程
    scaler_data = date.iloc[:, 0:3]
    scaler = StandardScaler()
    # scaler.fit(scaler_data)

    std_data = scaler.fit_transform(scaler_data)
    #生成新的数据集,对列名进行修改
    date1 = pd.DataFrame(std_data).join(date.iloc[:, 3:8])

    columns = date.columns
    columns_list = list(columns)

    rename_list = ['distance_from_home', 'distance_from_last_transaction',
                   'ratio_to_median_purchase_price',
                   'repeat_retailer',
                   'used_chip',
                   'used_pin_number',
                   'online_order',
                   'fraud']
    map_dict = {idx: name for idx, name in enumerate(rename_list)}
    # map_dict

    date1 = date1.rename(columns=map_dict)
    # print(date1)

    #指定自变量与因变量，分割训练集与测试集
    X = date1.drop(columns=['fraud'], axis=1)
    y = date1['fraud']
    y = y.astype(np.int64)

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.3, random_state=100)


    # 由于欺诈与未欺诈样本数量相差较大，需要处理非均衡样本
    # 使用SMOTE来对欺诈性数据进行抽样。
    # SMOTE将基于我们在原始数据集中已经拥有的欺诈数据，综合生成更多的欺诈数据样本
    #1)使用SMOTE方法
    train_x_SMOTE, train_y_SMOTE = SMOTE().fit_resample(train_x, train_y)


    #2）使用ADASYN 方法
    ada = ADASYN(random_state=42)
    train_x_ada, train_y_ada = ada.fit_resample(train_x, train_y)
    # print(train_y_ada.value_counts())
    # print(train_x, test_x, train_y, test_y, y, train_x_SMOTE, train_y_SMOTE, train_x_ada, train_y_ada)


    return train_x, test_x, train_y, test_y, y, train_x_SMOTE, train_y_SMOTE, train_x_ada, train_y_ada


    # y.value_counts()
    #
    # train_y.value_counts()
    # 修正了训练集标签的欺诈权重，测试集未做处理
    # 其原因为，测试集应反映真实情况

def date_visualization(c_path):
    date = pd.read_csv(c_path, engine='python')
    #显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # counts = date['fraud'].value_counts()
    # plt.figure(figsize=(2 * 7, 1 * 7))
    # ax = plt.subplot(1, 2, 1)
    # counts.plot(kind='pie', ax=ax, autopct='%.2f%%')
    # ax = plt.subplot(1, 2, 2)
    # counts.plot(kind='bar', ax=ax)
    # plt.savefig('./Pictures/非平衡数据下欺诈与非欺诈比例图')
    # plt.show()
    #
    # train_x, test_x, train_y, test_y, y = date_processing(c_path=c_path)
    #
    # plt.figure(figsize=(2 * 7, 1 * 7))
    # ax = plt.subplot(1, 2, 1)
    # train_y.plot(kind='bar', ax=ax, color='y', label='SMOTE采样后train_y欺诈与非欺诈比例')
    # ax = plt.subplot(1, 2, 2)
    # test_y.plot(kind='bar', ax=ax, color='c', label='SMOTE采样后test_y欺诈与非欺诈比例')
    # plt.savefig('./Pictures/SMOTE采样后欺诈与非欺诈比例图')
    # plt.show()
    #
    # #k可视化特征重要性
    # rfc = RandomForestClassifier()
    # rfc = rfc.fit(train_x, train_y)
    # feature_imp = pd.Series(rfc.feature_importances_,
    #                         index=train_x.columns).sort_values(ascending=False)
    # plt.figure(figsize=(15, 13))
    # sns.barplot(x=feature_imp, y=feature_imp.index)
    # plt.xlabel('features_importances')
    # plt.ylabel('features_names')
    # plt.title('features_importances of credict_fraud')
    # plt.savefig('./Pictures/features_importances of credict_fraud')
    # plt.show()







if __name__ == '__main__':
    import time

    start_time = time.time()

    date_processing(c_path='C:\\Users\\123\\Desktop\\2022年首届钉钉杯大学生大数据挑战赛初赛题目\\A题\\数据集\\card_transdata.csv')
    # date_visualization(c_path='C:\\Users\\123\\Desktop\\2022年首届钉钉杯大学生大数据挑战赛初赛题目\\A题\\数据集\\card_transdata.csv')