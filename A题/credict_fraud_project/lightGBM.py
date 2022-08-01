import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, f1_score, \
    matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, plot_importance
from sklearn.model_selection import GridSearchCV

from A题.credict_fraud_project.date_processing import date_processing


def lightgbm_building():
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
        lgbm = LGBMClassifier(n_estimators=100, boost_from_average=False)
        lgbm.fit(one_dict['X'], one_dict['Y'])
        lgbm_pred = lgbm.predict(test_x)
        one_dict['y_pred'] = lgbm_pred
        one_dict['model'] = lgbm
        n_errors = (lgbm_pred != test_y).sum()
        one_dict['预测错误个数'] = n_errors
        acc = accuracy_score(test_y, lgbm_pred)
        prec = precision_score(test_y, lgbm_pred)
        rec = recall_score(test_y, lgbm_pred)
        f1 = f1_score(test_y, lgbm_pred)
        MCC = matthews_corrcoef(test_y, lgbm_pred)
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



        # lgbm_pred = lgbm.predict(test_x)
        # lgbm_pred_proba = lgbm.predict_proba(test_x)[:, 1]
        # if str(i[0]) == 'train_x':
        #     print('非平衡数据下模型混淆矩阵为：', classification_report(test_y, lgbm_pred))
        #     n_errors = (lgbm_pred != test_y).sum()
        #     print(f'非平衡数据下模型预测错误个数为：{n_errors}')
        #     acc = accuracy_score(test_y, lgbm_pred)
        #     print("非平衡数据下准确率为： {}".format(acc))
        #     prec = precision_score(test_y, lgbm_pred)
        #     print("非平衡数据下精确率为：{}".format(prec))
        #     rec = recall_score(test_y, lgbm_pred)
        #     print("非平衡数据下召回率为：{}".format(rec))
        #     f1 = f1_score(test_y, lgbm_pred)
        #     print(" 非平衡数据下F1-Score 为： {}".format(f1))
        #     print('非平衡数据下混淆矩阵为：', classification_report(test_y, lgbm_pred))
        # if str(i[0]) == 'train_x_SMOTE':
        #     print('SMOTE后模型混淆矩阵为：', classification_report(test_y, lgbm_pred))
        #     n_errors = (lgbm_pred != test_y).sum()
        #     print(f'SMOTE后模型预测错误个数为：{n_errors}')
        #     acc = accuracy_score(test_y, lgbm_pred)
        #     print("SMOTE后准确率为： {}".format(acc))
        #     prec = precision_score(test_y, lgbm_pred)
        #     print("SMOTE后精确率为：{}".format(prec))
        #     rec = recall_score(test_y, lgbm_pred)
        #     print("SMOTE后召回率为：{}".format(rec))
        #     f1 = f1_score(test_y, lgbm_pred)
        #     print(" SMOTE后F1-Score 为： {}".format(f1))
        #     print('SMOTE后混淆矩阵为：', classification_report(test_y, lgbm_pred))
        # if str(i[0]) == 'train_x_ada':
        #     print(f'ADASYN后模型混淆矩阵为：', classification_report(test_y, lgbm_pred))
        #     n_errors = (lgbm_pred != test_y).sum()
        #     print(f'ADASYN后模型预测错误个数为：{n_errors}')
        #     acc = accuracy_score(test_y, lgbm_pred)
        #     print("ADASYN后准确率为： {}".format(acc))
        #     prec = precision_score(test_y, lgbm_pred)
        #     print("ADASYN后精确率为：{}".format(prec))
        #     rec = recall_score(test_y, lgbm_pred)
        #     print("ADASYN后召回率为：{}".format(rec))
        #     f1 = f1_score(test_y, lgbm_pred)
        #     print("ADASYN后F1-Score 为： {}".format(f1))
        #     print('ADASYN后混淆矩阵为：', classification_report(test_y, lgbm_pred))

    # lgbm.fit(train_x, train_y)
    # lgbm_pred = lgbm.predict(test_x)
    # lgbm_pred_proba = lgbm.predict_proba(test_x)[:, 1]
    # print(f'初始模型准确度:{accuracy_score(test_y, lgbm_pred) * 100}\n初始模型AUC值:{roc_auc_score(test_y, lgbm_pred_proba)}\n{classification_report(test_y, lgbm_pred)}')

    # params = {
    #     'max_depth': [6, 8, 10],
    #     'learning_rate': [0.2],
    #     'num_leaves': [10, 20, 30]
    # }
    # lgbm_gs = GridSearchCV(lgbm, cv=3, param_grid=params, n_jobs=-1)
    # lgbm_gs.fit(train_x, train_y)
    # best = lgbm_gs.best_estimator_
    # print('lightGBM最优参数为：', lgbm_gs.best_params_)
    # lgbm_gs_pred = best.predict(test_x)
    # lgbm_gs_pred_proba = best.predict_proba(test_x)[:, 1]
    #
    # print(
    #     f'最优模型准确度:{accuracy_score(test_y, lgbm_gs_pred) * 100}\n最优模型AUC值:{roc_auc_score(test_y, lgbm_gs_pred_proba)}')

    # 可视化混淆矩阵
    # print('最优模型混淆矩阵为：', classification_report(test_y, lgbm_gs_pred))
    #
    # n_errors = (lgbm_gs_pred != test_y).sum()
    # print(f'模型预测错误个数为：{n_errors}')
    #
    # acc = accuracy_score(test_y, lgbm_gs_pred)
    # print("准确率为： {}".format(acc))
    #
    # prec = precision_score(test_y, lgbm_gs_pred)
    # print("精确率为：{}".format(prec))
    #
    # rec = recall_score(test_y, lgbm_gs_pred)
    # print("召回率为：{}".format(rec))
    #
    # f1 = f1_score(test_y, lgbm_gs_pred)
    # print(" F1-Score 为： {}".format(f1))
    # print('混淆矩阵为：', classification_report(test_y, lgbm_gs_pred))
    #
    # cm = confusion_matrix(test_y, lgbm_gs_pred)
    # sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    # plt.title('LightGBM Confusion Matrix')
    # plt.savefig('LightGBM_con_matrix')
    # plt.show()
    #
    # #绘制roc曲线
    # LightGBM_roc_auc = roc_auc_score(test_y, lgbm_pred)
    # fpr, tpr, thresholds = roc_curve(test_y, lgbm_pred_proba)
    # plt.figure()
    # plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % LightGBM_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.legend(loc='lower right')
    # plt.savefig('LightGBM_roc_curve')
    # plt.show()


if __name__ == '__main__':
    import time

    start_time = time.time()
    lightgbm_building()
