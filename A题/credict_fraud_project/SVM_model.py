import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.metrics import sensitivity_specificity_support

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.svm import SVC

from date_processing import date_processing
from sklearn.model_selection import GridSearchCV, cross_val_score


def SVM_building():
    c_path = 'C:\\Users\\123\\Desktop\\2022年首届钉钉杯大学生大数据挑战赛初赛题目\\A题\\数据集\\card_transdata.csv'
    train_x, test_x, train_y, test_y, y = date_processing(c_path=c_path)
    svm_model = sklearn.svm.SVC() #实例化模型

    svm_params = {"C": np.arange(1, 10), "kernel": ["linear", "rbf"]} #设置网格搜索参数
    svm_cv_model = GridSearchCV(svm_model, svm_params, cv=7, n_jobs=-1, verbose=7).fit(train_x, train_y)
    print('svm最优得分：', svm_cv_model.best_score_)

    best_params = svm_cv_model.best_params_
    print('svm最优参数为：', best_params)
    svm = sklearn.svm.SVC(C=best_params['C'], kernel=best_params['kernel'], probability=True).fit(train_x,
                                                                                                  train_y)
    svm = svm.fit(train_x, train_y)
    y_pred_svm = svm.predict(test_x)
    #
    print('svm模型准确度：', accuracy_score(test_y, y_pred_svm))

    #进行交叉验证
    print('交叉验证均值：', cross_val_score(svm, test_x, test_y, cv=20).mean())

    #可视化混淆矩阵
    print('混淆矩阵为：', classification_report(test_y, y_pred_svm))
    cm = confusion_matrix(test_y, y_pred_svm)
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title('SVC Confusion Matrix')
    plt.savefig('svc_con_mat')
    plt.show()

    #绘制roc曲线
    svm_roc_auc = roc_auc_score(test_y, svm.predict(test_x))
    fpr, tpr, thresholds = roc_curve(test_y, svm.predict_proba(test_x)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % svm_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    SVM_building()