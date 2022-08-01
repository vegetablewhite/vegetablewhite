import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Recall = TP/(TP+FN)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, classification_report, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, KFold
# import itertoolsimport
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve
from A题.credict_fraud_project.date_processing import date_processing
from sklearn.metrics import confusion_matrix,classification_report


# def printing_Kfold_scores(train_x_data, train_y_data):
#     fold = KFold(len(train_y_data), 5, shuffle=False)
#
#     # Different C parameters
#     c_param_range = [0.01, 0.1, 1, 10, 100]
#
#     results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
#     results_table['C_parameter'] = c_param_range
#
#     # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
#     j = 0
#     for c_param in c_param_range:
#         print('-------------------------------------------')
#         print('C parameter: ', c_param)
#         print('-------------------------------------------')
#         print('')
#
#         recall_accs = []
#         for iteration, indices in enumerate(fold, start=1):
#             # Call the logistic regression model with a certain C parameter
#             lr = LogisticRegression(C=c_param, penalty='l1')
#
#             # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
#             # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
#             lr.fit(train_x_data.iloc[indices[0], :], train_y_data.iloc[indices[0], :].values.ravel())
#
#             # Predict values using the test indices in the training data
#             y_pred = lr.predict(train_x_data.iloc[indices[1], :].values)
#
#             # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
#             recall_acc = recall_score(train_y_data.iloc[indices[1], :].values, y_pred)
#             recall_accs.append(recall_acc)
#             print('Iteration ', iteration, ': recall score = ', recall_acc)
#
#         # The mean value of those recall scores is the metric we want to save and get hold of.
#         results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
#         j += 1
#         print('')
#         print('Mean recall score ', np.mean(recall_accs))
#         print('')
#
#     best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
#
#     # Finally, we can check which C parameter is the best amongst the chosen.
#     print('*********************************************************************************')
#     print('Best model to choose from cross validation is with C parameter = ', best_c)
#     print('*********************************************************************************')
#
#     return best_c

#  不需要自己定义混淆矩阵函数，调用sklearn中的confusion_matrix即可
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=0)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
#
def logit_building():
    c_path = 'C:\\Users\\123\\Desktop\\2022年首届钉钉杯大学生大数据挑战赛初赛题目\\A题\\数据集\\card_transdata.csv'
    train_x, test_x, train_y, test_y, y, train_x_SMOTE, train_y_SMOTE, train_x_ada, train_y_ada = date_processing(
        c_path=c_path)

    # best_c = printing_Kfold_scores(train_x_data=train_x, train_y_data=train_y)
    train_list = [
        {'X': train_x, 'Y': train_y, 'name': '原始数据'},
        {'X': train_x_SMOTE, 'Y': train_y_SMOTE, 'name': 'SMOTE后'},
        {'X': train_x_ada, 'Y': train_y_ada, 'name': 'ADASYN后'},
    ]

    final_dict_list = []
    for one_dict in train_list:
        name = one_dict['name']
        logit_clf = LogisticRegression()
        logit_clf.fit(one_dict['X'], one_dict['Y'])
        logit_clf_pred = logit_clf.predict(test_x)
        one_dict['y_pred'] = logit_clf_pred
        one_dict['model'] = logit_clf_pred
        n_errors = (logit_clf_pred != test_y).sum()
        one_dict['预测错误个数'] = n_errors
        acc = accuracy_score(test_y, logit_clf_pred)
        prec = precision_score(test_y, logit_clf_pred)
        rec = recall_score(test_y, logit_clf_pred)
        f1 = f1_score(test_y, logit_clf_pred)
        MCC = matthews_corrcoef(test_y, logit_clf_pred)
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

    # best_lr = LogisticRegression(C=best_c, penalty='l1').fit(train_x, train_y)
    # pred_y = best_lr.predict(test_x)
    #
    # print(f'模型最佳参数为：{best_c}')
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

    # Matthews相关系数在机器学习中用作衡量二进制和多类分类质量的指标。它考虑了真假正例和负例，
    # 通常被认为是平衡的度量，即使类别的大小差异很大，也可以使用该度量。MCC本质上是介于-1和+1之间的相关系数值。
    # 系数+1代表理想预测，0代表平均随机预测，-1代表逆预测
    # MCC = matthews_corrcoef(test_y, pred_y)
    # print("Matthews相关系数为：{}".format(MCC))
    #
    # #不需要自己定义混淆矩阵函数，调用confusion_matrix即可
    # cm = confusion_matrix(test_y, pred_y)
    # sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    # plt.title('Logit Confusion Matrix')
    # plt.savefig('./Pictures/Logit_con_matrix')
    # plt.show()
    #
    # # 绘制roc曲线
    # Logit_roc_auc = roc_auc_score(test_y, lgbm_pred)
    # fpr, tpr, thresholds = roc_curve(test_y, lgbm_pred_proba)
    # plt.figure()
    # plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % Logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.legend(loc='lower right')
    # plt.savefig('./Pictures/Logit_roc_curve')
    # plt.show()

if __name__ == '__main__':
    logit_building()






