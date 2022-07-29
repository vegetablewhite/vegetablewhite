import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, f1_score, \
    matthews_corrcoef
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from A题.credict_fraud_project.date_processing import date_processing

def random_forest_building():
    c_path = 'C:\\Users\\123\\Desktop\\2022年首届钉钉杯大学生大数据挑战赛初赛题目\\A题\\数据集\\card_transdata.csv'
    train_x, test_x, train_y, test_y, y = date_processing(c_path=c_path)


    # random forest model creation
    rfc = RandomForestClassifier()

    rfc_params = {'n_estimators': [100, 200, 500],
                  'max_features': [3, 5, 7],
                  'min_samples_split': [5, 10, 20]}

    rf_cv_model = GridSearchCV(rfc, rfc_params, cv=5, n_jobs=-1, verbose=1).fit(train_x, train_y)

    best_params = rf_cv_model.best_params_
    print(f'模型最佳参数为：{best_params}')

    rfc = RandomForestClassifier(max_features=best_params['max_features'],
                                min_samples_split=best_params['min_samples_split'],
                                n_estimators=best_params['n_estimators']).fit(train_x, train_y)
    # rfc = rfc.fit(train_x, train_y)
    pred_y = rfc.predict(test_x)

    n_errors = (pred_y != test_y).sum()
    print(f'模型预测错误个数为：{n_errors}')

    acc = accuracy_score(test_y, pred_y)
    print("准确率为： {}".format(acc))

    prec = precision_score(test_y, pred_y)
    print("精确率为：{}".format(prec))

    rec = recall_score(test_y, pred_y)
    print("召回率为：{}".format(rec))

    f1 = f1_score(test_y, pred_y)
    print(" F1-Score 为： {}".format(f1))

    # Matthews相关系数在机器学习中用作衡量二进制和多类分类质量的指标。它考虑了真假正例和负例，
    # 通常被认为是平衡的度量，即使类别的大小差异很大，也可以使用该度量。MCC本质上是介于-1和+1之间的相关系数值。
    # 系数+1代表理想预测，0代表平均随机预测，-1代表逆预测
    MCC = matthews_corrcoef(test_y, pred_y)
    print("Matthews相关系数为：{}".format(MCC))
    print('混淆矩阵为：', classification_report(test_y, pred_y))

    # # 可视化features重要性
    # feature_imp = pd.Series(rfc.feature_importances_,
    #                         index=train_x.columns).sort_values(ascending=False)
    # plt.figure(figsize=(15, 13))
    # sns.barplot(x=feature_imp, y=feature_imp.index)
    # plt.xlabel('features_importances')
    # plt.ylabel('features_names')
    # plt.title('features_importances of credict_fraud')
    # plt.savefig('./Pictures/features_importances of credict_fraud')
    # plt.show()

    cm = confusion_matrix(test_y, pred_y)
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title('RFC Confusion Matrix')
    plt.savefig('RFC_con_matrix')
    plt.show()

    rfc_roc_auc = roc_auc_score(test_y, pred_y)
    fpr, tpr, thresholds = roc_curve(test_y, pred_y)
    plt.figure()
    plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % rfc_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.savefig('RFC_ROC_curve')
    plt.show()



if __name__ == '__main__':
    random_forest_building()