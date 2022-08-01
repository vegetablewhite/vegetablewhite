import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, f1_score, \
    matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report


from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier

from A题.credict_fraud_project.date_processing import date_processing



def XGboost_building():
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
        xgb_clf = XGBClassifier()
        xgb_clf.fit(one_dict['X'], one_dict['Y'])
        xgb_clf_pred = xgb_clf.predict(test_x)
        one_dict['y_pred'] = xgb_clf_pred
        one_dict['model'] = xgb_clf
        n_errors = (xgb_clf_pred != test_y).sum()
        one_dict['预测错误个数'] = n_errors
        acc = accuracy_score(test_y, xgb_clf_pred)
        prec = precision_score(test_y, xgb_clf_pred)
        rec = recall_score(test_y, xgb_clf_pred)
        f1 = f1_score(test_y, xgb_clf_pred)
        MCC = matthews_corrcoef(test_y, xgb_clf_pred)
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
    #调参过程————由于准确率较高，未进行调参

    # dtrain = xgb.DMatrix(train_x, label=train_y)
    # dtest = xgb.DMatrix(test_x)
    # watchlist = [(dtrain, 'train')]
    #
    # # cv_params = {'max_depth': [1, 2, 3, 4, 5, 6],
    # #              'min_child_weight': [1, 2, 3, 4],
    # #              'subsample': [0.8, 0.9, 1], 'max_delta_step': [0, 1, 2, 4]}
    # # fix_params = {'learning_rate': 0.2, 'n_estimators': 100,
    # #               'objective': 'binary:logistic'}
    # # xgb_cv = GridSearchCV(xgb.XGBClassifier(**fix_params), cv_params, scoring='f1', cv=5)
    # # xgb_cv.fit(train_x, train_y)
    # # params = xgb_cv.best_params_
    #
    # params = {'booster': 'gbtree',
    #           'objective': 'binary:logistic',
    #           'eval_metric': 'auc',
    #           'max_depth': 5,
    #           'lambda': 10,
    #           'subsample': 0.75,
    #           'colsample_bytree': 0.75,
    #           'min_child_weight': 2,
    #           'eta': 0.025,
    #           'seed': 0,
    #           'nthread': 8,
    #           'gamma': 0.15,
    #           'learning_rate': 0.01}
    #
    # bst = xgb.train(params, dtrain, num_boost_round=20, evals=watchlist)
    # pred_y = bst.predict(dtest)

    # xgb_clf = XGBClassifier()
    # train = [train_x, train_y]
    # eval = [test_x, test_y]
    # xgb_clf = xgb_clf.fit(train_x, train_y, eval_metric=['logloss', 'auc', 'error'], eval_set=[train, eval])
    #
    # pred_y = xgb_clf.predict(test_x)
    #
    # n_errors = (pred_y != test_y).sum()
    # print(f'模型预测错误个数为：{n_errors}')
    #
    # acc = accuracy_score(test_y, pred_y)
    # print("准确率为： {}".format(acc))
    #
    # prec = precision_score(test_y, pred_y)
    # print("精确率为：{}".format(prec))
    #
    # rec = recall_score(test_y, pred_y)
    # print("召回率为：{}".format(rec))
    #
    # f1 = f1_score(test_y, pred_y)
    # print(" F1-Score 为： {}".format(f1))
    # print('混淆矩阵为：', classification_report(test_y, pred_y))
    #
    # cm = confusion_matrix(test_y, pred_y)
    # sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap='GnBu')
    # plt.title('Xgboost Confusion Matrix')
    # plt.savefig('Xgboost_con_matrix')
    # plt.show()
    #
    # # 绘制roc曲线
    # XGboost_roc_auc = roc_auc_score(test_y, pred_y)
    #
    # fpr, tpr, thresholds = roc_curve(test_y, pred_y)
    # plt.figure()
    # plt.plot(fpr, tpr)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.title("roc_curve of %s(AUC=%.4f)" % ('Xgbc', XGboost_roc_auc))
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.savefig('Xgboost_AUC_curve')
    # plt.show()

if __name__ == '__main__':
    XGboost_building()


