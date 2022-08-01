import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

from A题.credict_fraud_project.date_processing import date_processing
def kmeans_building():
    c_path = 'C:\\Users\\123\\Desktop\\2022年首届钉钉杯大学生大数据挑战赛初赛题目\\A题\\数据集\\card_transdata.csv'
    train_x, test_x, train_y, test_y, y, train_x_SMOTE, train_y_SMOTE, train_x_ada, train_y_ada = date_processing(
        c_path=c_path)
    train_list = [
        {'X': train_x, 'Y': train_y, 'name': '原始数据'},
        {'X': train_x_SMOTE, 'Y': train_y_SMOTE, 'name': 'SMOTE后'},
        {'X': train_x_ada, 'Y': train_y_ada, 'name': 'ADASYN后'},
    ]

    final_dict_list = []
    for one_dict in train_list:
        name = one_dict['name']
        km_model = KMeans(n_clusters = 2, random_state=10)
        km_model.fit(one_dict['X'], one_dict['Y'])
        km_pred = km_model.predict(test_x)
        one_dict['y_pred'] = km_pred
        one_dict['model'] = km_pred
        n_errors = (km_pred != test_y).sum()
        one_dict['预测错误个数'] = n_errors
        acc = accuracy_score(test_y, km_pred)
        prec = precision_score(test_y, km_pred)
        rec = recall_score(test_y, km_pred)
        f1 = f1_score(test_y, km_pred)
        MCC = matthews_corrcoef(test_y, km_pred)
        one_final_dict = {
            'name': one_dict['name'],
            '预测错误个数': n_errors,
            '准确率': acc,
            '精确度': prec,
            '召回率': rec,
            'F1-Score': f1,
            'Matthews相关系数': MCC
        }

        final_dict_list.append(one_final_dict)
    print(final_dict_list)
    return final_dict_list

    # km= KMeans(n_clusters = 3, random_state=10).fit( train_x, train_y)
    # print(km)
    # centers=km.cluster_centers_
    # print(centers)
    # pred_y=km.predict(test_x)
    # print(pred_y)
    # colors = ['red', 'green', 'blue']
    # fig,ax1=plt.subplots(1)
    # for i in range(3):
    #     ax1.scatter(train_x[train_y==i,0],train_x[train_y==i,1],s=8,c=color[i])
    # plt.show()

    #centroids, cluster = kmeans(list(dataset), 2)
    #print('质心为：%s' % centroids)
    #print('集群为：%s' % cluster)
    #for i in range(len(dataset)):
      #  plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='green', s=40, label='原始点')
       # #  记号形状       颜色      点的大小      设置标签
        #for j in range(len(centroids)):
         #   plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='质心')
          #  plt.show()
if __name__=='__main__':
    kmeans_building()
