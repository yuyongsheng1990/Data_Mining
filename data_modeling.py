# _*_ coding: utf-8 _*_
# @Time: 2021/10/716:11 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description: 他克莫司DM项目学习：3. 他克莫司初始&调整方案建模

import pandas as pd
import numpy as np

import re
import os
import sys

from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model

# Tell auto_ml which column is 'output'
# Also note columns that aren't purely numerical
# Examples include ['nlp', 'date', 'categorical', 'ignore']

project_path = os.getcwd()

# 判断文件路径是否存在，如果不存在则创建该路径
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

# 体重分段
def dispersed_weight(x, y):
    if x <= 50 and y == 0:
        z = '偏低'
    elif x >= 90 and y == 0:
        z = '偏高'
    elif x <= 40 and y == 1:
        z = '偏低'
    elif x >= 80 and y == 1:
        z = '偏高'
    else:
        z = '中等'
    return z

# 身高分段
def dispersed_height(x, y):
    if x <= 160 and y == 0:
        z = '偏低'
    elif x >= 175 and y == 0:
        z = '偏高'
    elif x <= 150 and y == 1:
        z = '偏低'
    elif x >= 165 and y == 1:
        z = '偏高'
    else:
        z = '中等'
    return z

'''
def rgb2hex(rgbcolor):
    r, g, b = rgbcolor
    return (r << 16) + (g << 8) + b
'''

def main():
    print('--------------------读取数据------------------------------')
    df_initial = pd.read_excel(project_path + '/data/data_from_mysql/df_初始方案建模数据集.xlsx')
    df_adjust = pd.read_excel(project_path + '/data/data_from_mysql/df_调整方案建模数据集.xlsx')

    df_initial = df_initial.drop(['Unnamed: 0'], axis=1)
    df_adjust = df_adjust.drop(['Unnamed: 0'], axis=1)

    print(df_initial.columns)
    print(df_initial.shape)
    print(df_adjust.columns)
    print(df_adjust.shape)

    df = pd.concat([df_initial, df_adjust], axis=0)
    df = df.reset_index()
    del df['index']
    print(df.shape)

    ## 初始用药中没有康唑类药物，因此合并后缺失值插补0
    df['康唑类'] = df['康唑类'].replace(np.nan, 0)
    print(df.columns)

    ## 将五酯软胶囊修改为0-1变量
    df['五酯软胶囊'] = df['五酯软胶囊日剂量'].apply(lambda x: 1 if x > 0 else 0)
    print(df[df['五酯软胶囊'] == 1].shape)

    df = df[['PATIENT_ID', '日剂量', 'FREQUENCY', 'TDM检测结果', '上一次TDM值', '上一次日剂量', \
             '移植术后天数', '是否亲属活体', '性别', '年龄', '身高', '体重', 'BMI', 'CYP3A5', 'ABCB1', \
             '五酯软胶囊', '康唑类', '钙通道阻滞剂', '拉唑类', '酶诱导剂', 'MPA类药物', '糖皮质激素日剂量', \
             '中性粒细胞百分数', '尿酸', '总胆红素', '淋巴细胞百分数', '红细胞压积', '红细胞计数', '血红蛋白', '肌酐', \
             '谷草转氨酶', '高血压', '糖尿病', '生理病理状况']]

    print(df.shape)
    print(df['淋巴细胞百分数'].describe())

    print(df[df['ABCB1'] == 0].shape[0], df[df['ABCB1'] == 0].shape[0] / df.shape[0])
    print(df[df['ABCB1'] == 1].shape[0], df[df['ABCB1'] == 1].shape[0] / df.shape[0])
    print(df[df['ABCB1'] == 2].shape[0], df[df['ABCB1'] == 2].shape[0] / df.shape[0])

    print(df_adjust['性别'].isnull().sum())

    for i in df.columns:
        df[i] = df[i].astype('float')
        if df[i].isnull().sum() > 0:
            print(i, df[i].isnull().sum())

    print(len(df['PATIENT_ID'].unique()))

    from sklearn.model_selection import train_test_split
    print('-------------------------划分数据集---------------------------')
    # 划分训练集和测试集，比例为8:2
    x = df.iloc[:, 1:]
    y = df['日剂量']
    tran_x, test_x, tran_y, test_y = train_test_split(x, y, test_size=0.2, random_state=5)

    tran_x = tran_x.drop(['肌酐', 'TDM检测结果', 'FREQUENCY'], axis=1)
    test_x = test_x.drop(['肌酐', 'TDM检测结果', 'FREQUENCY'], axis=1)

    print(tran_x.columns)

    column_descriptions = { '日剂量': 'output'}
    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
    ml_predictor.train(tran_x, model_names=['XGBRegressor'])

    # Score the model on test data
    test_score = ml_predictor.score(test_x, test_x['日剂量'])

    # auto_ml is specifically tuned for running in production
    # It can get predictions on an individual row (passed in as a dictionary)
    # A single prediction like this takes ~1 millisecond
    # Here we will demonstrate saving the trained model, and loading it again

    file_name = ml_predictor.save()
    trained_model = load_ml_model(file_name)

    # .predict and .predict_proba take in either:
    # A pandas DataFrame

    # A list of dictionaries
    # A single dictionary (optimized for speed in production evironments)
    predictions = ml_predictor.predict(test_x)
    print(predictions)

    predictions = trained_model.predict(test_x)
    print(predictions)

    print(df.columns)


    # 重要变量数值离散化，分段
    print('-----------------------------重要变量数值离散化，分段----------------------------------')
    df_IMP = df[['日剂量', '性别', '年龄', '身高', '体重', '移植术后天数', '上一次日剂量', '上一次TDM值', '红细胞压积', \
                 '尿酸', '中性粒细胞百分数', '总胆红素', '淋巴细胞百分数', '糖皮质激素日剂量', \
                 'BMI', '谷草转氨酶', '高血压', '拉唑类', 'ABCB1', 'CYP3A5', '酶诱导剂', 'MPA类药物']]

    print(df_IMP.shape)
    print(df_IMP['糖皮质激素日剂量'].describe())

    df_IMP['CYP3A5'] = df_IMP['CYP3A5'].apply(lambda x: 'AA' if x == 0 else 'AG' if x == 1 else 'GG' if x == 2 else x)
    df_IMP['ABCB1'] = df_IMP['ABCB1'].apply(lambda x: 'CC' if x == 0 else 'CT' if x == 1 else 'TT' if x == 2 else x)

    print(np.unique(df_IMP['ABCB1'].astype('str')))

    ## 性别0是男，1是女
    df_IMP[df_IMP['性别'] == 1]['身高'].describe()



    for i in range(df_IMP.shape[0]):
        df_IMP.loc[i, '体重_分段'] = dispersed_weight(df_IMP.loc[i, '体重'], df_IMP.loc[i, '性别'])


    for i in range(df_IMP.shape[0]):
        df_IMP.loc[i, '身高_分段'] = dispersed_height(df_IMP.loc[i, '身高'], df_IMP.loc[i, '性别'])

    # 变量分段
    df_IMP['年龄_分段'] = df_IMP['年龄'].apply(
        lambda x: '小于18岁' if x < 18 else '18到60岁' if 18 <= x <= 60 else '大于60岁' if x > 60 else x)
    df_IMP['移植术后天数_分段'] = df_IMP['移植术后天数'].apply(
        lambda x: '不超过3天' if x <= 3 else '3到15天' if 3 < x <= 15 else '大于15天' if x > 15 else x)
    # df_IMP['体重_分段']=df_IMP['体重'].apply(lambda x: '不超过50kg' if x<=50 else '50到65kg' if 50<x<=65 else '大于65kg' if x>65 else x)
    # df_IMP['身高_分段']=df_IMP['身高'].apply(lambda x: '不超过160cm' if x<=160 else '160到173cm' if 160<x<=173 else '大于173cm' if x>173 else x)
    df_IMP['上一次日剂量'] = df_IMP['上一次日剂量'].apply(
        lambda x: '小于4mg' if x < 4 else '4到6mg' if 4 <= x <= 6 else '大于6mg' if x > 6 else x)
    df_IMP['上一次TDM值'] = df_IMP['上一次TDM值'].apply(
        lambda x: '小于8ng/ml' if x < 8 else '8到15ng/ml' if 8 <= x <= 15 else '大于15ng/ml' if x > 15 else x)
    df_IMP['红细胞压积'] = df_IMP['红细胞压积'].apply(
        lambda x: '偏低' if x < 0.37 else np.nan if 0.37 <= x <= 0.49 else '偏高' if x > 0.49 else x)
    # df_IMP['五酯软胶囊日剂量']=df_IMP['五酯软胶囊日剂量'].apply(lambda x:  '未使用' if x==0 else '不超过1g' if 0<x<=1 else '大于1g' if x>1 else x)
    df_IMP['糖皮质激素日剂量'] = df_IMP['糖皮质激素日剂量'].apply(
        lambda x: '小于12mg' if x <= 12 else '12到25mg' if 12 < x <= 25 else '大于25mg' if x > 25 else x)
    df_IMP['尿酸'] = df_IMP['尿酸'].apply(lambda x: '偏低' if x < 89 else np.nan if 89 <= x <= 416 else '偏高' if x > 416 else x)
    df_IMP['中性粒细胞百分数'] = df_IMP['中性粒细胞百分数'].apply(
        lambda x: '偏低' if x < 42.3 else np.nan if 42.3 <= x <= 71.5 else '偏高' if x > 71.5 else x)

    ## 逻辑回归没有用BMI
    df_IMP['BMI'] = df_IMP['BMI'].apply(
        lambda x: '偏低' if x < 18.5 else np.nan if 18.5 <= x <= 24 else '偏高' if x > 24 else x)
    df_IMP['总胆红素'] = df_IMP['总胆红素'].apply(
        lambda x: '偏低' if x < 3.4 else np.nan if 3.4 <= x <= 17.1 else '偏高' if x > 17.1 else x)
    df_IMP['谷草转氨酶'] = df_IMP['谷草转氨酶'].apply(lambda x: np.nan if x <= 40 else '偏高' if x > 40 else x)
    # df_IMP['血红蛋白']=df_IMP['血红蛋白'].apply(lambda x:'偏低' if x<115 else np.nan if 115<=x<=175 else '偏高' if x>175 else x)
    df_IMP['淋巴细胞百分数'] = df_IMP['淋巴细胞百分数'].apply(
        lambda x: '偏低' if x < 20 else np.nan if 20 <= x <= 40 else '偏高' if x > 40 else x)
    # df_IMP['总胆汁酸']=df_IMP['总胆汁酸'].apply(lambda x: np.nan if x<=10 else '偏高' if x>10 else x)
    # df_IMP['直接胆红素']=df_IMP['直接胆红素'].apply(lambda x: np.nan if x<=6.8 else '偏高' if x>6.8 else x)
    # df_IMP['淋巴细胞']=df_IMP['淋巴细胞'].apply(lambda x:'偏低' if x<0.8 else np.nan if 0.8<=x<=3.5 else '偏高' if x>3.5 else x)
    # df_IMP['红细胞']=df_IMP['红细胞'].apply(lambda x:'偏低' if x<3.5 else np.nan if 3.5<=x<=5.5 else '偏高' if x>5.5 else x)


    one_hot_cols = ['年龄_分段', '移植术后天数_分段', '体重_分段', '身高_分段', '上一次日剂量', '上一次TDM值', '红细胞压积', \
                    '糖皮质激素日剂量', '尿酸', '中性粒细胞百分数', 'BMI', '总胆红素', '谷草转氨酶', '淋巴细胞百分数', 'CYP3A5', 'ABCB1']
    for j in one_hot_cols:
        temp1 = pd.get_dummies(df_IMP[j])  # get_dummies进行one-hot编码
        for i in temp1.columns:
            df_IMP[str(j) + '_' + str(i)] = temp1[i]
    df_IMP = df_IMP.drop(one_hot_cols, axis=1)

    print(df_IMP.head())
    print(df_IMP.shape)


    ## 删除缺失值的样本
    df_IMP = df_IMP.reset_index()
    del df_IMP['index']
    missing_ratio = []
    for i in range(df_IMP.shape[0]):
        count = df_IMP.iloc[i, 1:].isnull().sum()
        missing_ratio.append(count)

    missing = pd.DataFrame()
    missing['ratio'] = missing_ratio
    index = missing[missing['ratio'] > 0].index
    df_IMP = df_IMP.drop(index, axis=0)
    df_IMP = df_IMP.reset_index()
    del df_IMP['index']

    print(df_IMP.shape)

    writer = pd.ExcelWriter(project_path + '/data/data_from_mysql/df_IMP_做倾向性评分_个体化用药.xlsx')
    df_IMP.to_excel(writer)
    writer.save()

    for i in df_IMP.columns:
        print(i, df_IMP[df_IMP[i] == 1].shape[0])
        print(df_IMP[df_IMP[i] == 0].shape[0])


    print(tran_x.columns)
    ## 根据重要变量再次建模
    print('-------------------------根据重要变量再次建模------------------------------')
    df_IMP_1 = df[['日剂量', '年龄', '身高', '体重', '移植术后天数', '上一次TDM值', '上一次日剂量', 'CYP3A5', '糖皮质激素日剂量', '红细胞压积', 'MPA类药物']]
    print(df_IMP_1.shape)
    print(tran_x.shape)
    print(test_x.shape)

    tran_x_1 = tran_x[['日剂量', '年龄', '身高', '体重', '移植术后天数', '上一次TDM值', '上一次日剂量', 'CYP3A5', '糖皮质激素日剂量', '红细胞压积', 'MPA类药物']]
    test_x_1 = test_x[['日剂量', '年龄', '身高', '体重', '移植术后天数', '上一次TDM值', '上一次日剂量', 'CYP3A5', '糖皮质激素日剂量', '红细胞压积', 'MPA类药物']]

    tran_x_1['group'] = 0
    test_x_1['group'] = 1
    tran_test = pd.concat([tran_x_1, test_x_1], axis=0)
    tran_test = tran_test.reset_index()
    del tran_test['index']
    print(tran_test.shape)


    ## 各个变量的描述统计（均值标准差，中位数上下四分位数，频数百分比）
    print(tran_x_1['红细胞压积'].describe())
    ## 各个变量的描述统计（均值标准差，中位数上下四分位数，频数百分比）
    # tran_x['TBIL(umol/l)'].describe()
    tran_x_1['MPA类药物'] = tran_x_1['MPA类药物'].astype('str')
    for i in np.unique(tran_x_1['MPA类药物'].astype('str')):
        print(i)
        print(tran_x_1[tran_x_1['MPA类药物'] == i].shape, tran_x_1[tran_x_1['MPA类药物'] == i].shape[0] / 2605)

    ## mann-whitney u检验
    print('----------------------mann-whitney u检验-------------------------')
    import scipy.stats as stats

    group1 = list(tran_x_1['红细胞压积'])
    group2 = list(test_x_1['红细胞压积'])

    u_statistic, pVal = stats.mannwhitneyu(group1, group2)
    print(u_statistic, pVal)


    ## 卡方检验
    print('----------------------卡方检验-------------------------')
    from scipy.stats import chi2_contingency

    r1 = []
    r2 = []
    tran_test['MPA类药物'] = tran_test['MPA类药物'].astype('str')
    for i in range(len(np.unique(tran_test['MPA类药物']))):
        r1.append(
            tran_test[(tran_test['group'] == 0) & (tran_test['MPA类药物'] == np.unique(tran_test['MPA类药物'])[i])].shape[0])
        r2.append(
            tran_test[(tran_test['group'] == 1) & (tran_test['MPA类药物'] == np.unique(tran_test['MPA类药物'])[i])].shape[0])

    abcd = np.array([r1, r2])
    print(abcd)
    result = chi2_contingency(abcd)
    print(result)

    tran_x_1 = tran_x_1.drop(['group'], axis=1)
    test_x_1 = test_x_1.drop(['group'], axis=1)

    print(tran_x_1.columns)


    column_descriptions = {'日剂量': 'output'}
    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
    ml_predictor.train(tran_x_1, model_names=['XGBRegressor'])

    # Score the model on test data
    test_score = ml_predictor.score(test_x_1, test_x_1['日剂量'])

    # auto_ml is specifically tuned for running in production
    # It can get predictions on an individual row (passed in as a dictionary)
    # A single prediction like this takes ~1 millisecond
    # Here we will demonstrate saving the trained model, and loading it again
    file_name = ml_predictor.save()

    trained_model = load_ml_model(file_name)

    # .predict and .predict_proba take in either:
    # A pandas DataFrame

    # A list of dictionaries
    # A single dictionary (optimized for speed in production evironments)
    predictions = ml_predictor.predict(test_x_1)

    print(predictions)
    print(test_x_1.shape)


    ## 真实值和预测值对比折线图
    print('----------------------真实值和预测值对比折线图--------------------------------')
    test_x_1['日剂量预测值'] = predictions
    test_x_1 = test_x_1.sort_values(['日剂量'], ascending=1)

    test_x_1['日剂量分组'] = test_x_1['日剂量'].apply(lambda x: 0 if x <= 4 else 1 if 4 < x <= 6 else 2 if x > 6 else x)
    print(np.unique(test_x_1['日剂量分组']))

    test_x_1['预测准确率'] = (test_x_1['日剂量预测值'] - test_x_1['日剂量']) / test_x_1['日剂量']
    test_x_1['预测准确率'] = test_x_1['预测准确率'].apply(lambda x: 1 if abs(x) <= 0.2 else 0)  # abs()是绝对值函数

    aaa = test_x_1[test_x_1['日剂量分组'] == 0]
    bbb = test_x_1[test_x_1['日剂量分组'] == 1]
    ccc = test_x_1[test_x_1['日剂量分组'] == 2]

    print(aaa.shape[0])
    print(bbb.shape[0])
    print(ccc.shape[0])

    print(aaa[aaa['预测准确率'] == 1].shape[0] / aaa.shape[0])
    print(bbb[bbb['预测准确率'] == 1].shape[0] / bbb.shape[0])
    print(ccc[ccc['预测准确率'] == 1].shape[0] / ccc.shape[0])

    from pylab import mpl

    mpl.rcParams['font.sans-serif'] = ['SimHei']  ##绘图显示中文
    mpl.rcParams['axes.unicode_minus'] = False

    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib import rc

    rc('mathtext', default='regular')



    sub_axix = filter(lambda x: x % 200 == 0, list(range(test_x_1.shape[0])))
    # plt.title('测试集他克莫司日剂量真实值和预测值对比折线图')
    plt.plot(list(range(test_x_1.shape[0])), np.array(list(test_x_1['日剂量预测值'])),
             color=(0.32941176470588235, 0.7294117647058823, 0.7490196078431373),
             label='The Predictive Value of Tac Daily Dose(mg)')
    # plt.plot(sub_axix, test_acys, color='red', label='testing accuracy')
    plt.plot(list(range(test_x_1.shape[0])), np.array(list(test_x_1['日剂量'])), color='indianred',
             label='The True Value of Tac Daily Dose(mg)')
    # plt.plot(list(range(test_x_1.shape[0])), thresholds, color='blue', label='threshold')
    plt.legend(bbox_to_anchor=(1.1, 1))  # 显示图例

    plt.xlabel('Number of Events(unit)')
    plt.ylabel('Tac Daily Dose(mg)')
    # plt.show()
    jpg_path = project_path + "/jpg/他克莫司_箱线图（python导出）"
    # 判断该路径是否存在，否则创建
    mkdir(jpg_path)
    plt.savefig(jpg_path + "/测试集折线图.jpg", dpi=300)

    '''缺少数据
    ## 重要性得分柱形图
    df_importance_21 = pd.read_excel(project_path + '/data/data_from_mysql/重要性得分.xlsx')
    print(df_importance_21.columns)
    print(df_importance_21)
    
    names = list(df_importance_21['Variable Name'])
    index = np.arange(len(names))
    plt.figure(figsize=(15, 8))
    plt.bar(index, df_importance_21['Importance Sore'], width=0.7,
            color=(0.32941176470588235, 0.7294117647058823, 0.7490196078431373), tick_label=names)
    plt.xticks(rotation=75)
    plt.ylabel('Importance Score')
    plt.xlabel('Variable Name')
    for a, b in zip(index, df_importance_21['Importance Sore']):
        plt.text(a, b + 0.002, '%.4f' % b, ha='center', va='bottom', fontsize=10)
    # plt.title('重要变量得分柱形图')
    plt.show()
    
    
    ##倾向性评分匹配后的数据集，生成箱线图
    df_ps_box = pd.read_excel(project_path + '/data/data_from_mysql/糖皮质激素日剂量12到25mg.xlsx')
    
    print(df_ps_box.columns)
    print(df_ps_box.shape)
    
    import seaborn as sns
    
    sns.set(style="whitegrid")
    df_ps_box = df_ps_box.rename(
        columns={'糖皮质激素日剂量_12到25mg': 'Daily dose glucocorticoid_12 to 25mg', '他克莫司日剂量': 'Tac Daily Dose(mg)'})
    ax = sns.boxplot(x="Daily dose glucocorticoid_12 to 25mg", y="Tac Daily Dose(mg)", data=df_ps_box,
                     color=(0.32941176470588235, 0.7294117647058823, 0.7490196078431373))
    
    fig = ax.get_figure()
    fig.savefig(jpg_path/Daily dose glucocorticoid_12 to 25mg.jpg", dpi=300)
    
    ## 验证集结果
    '''
    ## 读取验证集数据
    print('--------------------------验证集需求------------------------------------')
    df_val = pd.read_excel(project_path + '/data/test_dataset/他克莫司验证集需求20190826.xlsx')

    print(df_val.columns)
    print(df_val.shape)
    print(len(df_val['病人ID'].unique()))

    df_val = df_val.rename(columns={'GYP3A5*3': 'CYP3A5'})

    ## 移植术后时间如果不是首次用药则-3
    df_val = df_val.reset_index()
    del df_val['index']
    for i in range(df_val.shape[0]):
        if (df_val.loc[i, '是否首次用药'] == 0) | (df_val.loc[i, '移植术后天数'] >= 3):
            df_val.loc[i, '移植术后天数'] = df_val.loc[i, '移植术后天数'] - 3

    print(df_val['CYP3A5'].unique())

    df_val['CYP3A5'] = df_val['CYP3A5'].apply(
        lambda x: 0 if 'AA' in x else 1 if ('AG' in x) | ('GA' in x) else 2 if 'GG' in x else np.nan)
    # df_val_1['糖皮质激素日剂量']=df_val_1['糖皮质激素日剂量'].apply(lambda x: x.replace('g',''))

    df_val['糖皮质激素日剂量'] = df_val['糖皮质激素日剂量'].apply(lambda x: x.replace(' ', ''))

    ## 提取糖皮质激素日剂量，其中泼尼松的剂量*1.25转换成等量甲泼尼龙的剂量

    df_val['糖皮质激素日剂量'] = df_val['糖皮质激素日剂量'].apply(
        lambda x: float(re.findall(r'\d*\.?\d+', x)[0]) * 1.25 if '泼尼松' in x else float(
            re.findall(r'\d*\.?\d+', x)[0]) if '甲泼尼龙' in x else 0)
    print(df_val['MPA类药物'].unique())

    df_val['MPA类药物'] = df_val['MPA类药物'].apply(lambda x: 0 if '无' in x else 1)
    print(df_val[df_val['MPA类药物'] == 0].shape)

    df_val_1 = df_val[list(tran_x_1.columns)]
    print(df_val_1.head())
    print(tran_x_1.head())


    column_descriptions = {'日剂量': 'output'}
    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
    ml_predictor.train(tran_x_1, model_names=['XGBRegressor'])

    # Score the model on test data
    test_score = ml_predictor.score(df_val_1, df_val_1['日剂量'])

    # auto_ml is specifically tuned for running in production
    # It can get predictions on an individual row (passed in as a dictionary)
    # A single prediction like this takes ~1 millisecond
    # Here we will demonstrate saving the trained model, and loading it again

    file_name = ml_predictor.save()
    trained_model = load_ml_model(file_name)

    # .predict and .predict_proba take in either:
    # A pandas DataFrame

    # A list of dictionaries
    # A single dictionary (optimized for speed in production evironments)
    predictions = ml_predictor.predict(df_val_1)
    print(predictions)

    ## 真实值和预测值对比折线图
    print('-------------------------验证集真实值和预测值对比折线图-------------------------------')
    df_val_1['日剂量预测值'] = predictions
    df_val_1 = df_val_1.sort_values(['日剂量'], ascending=1)
    df_val_1['日剂量分组'] = df_val_1['日剂量'].apply(lambda x: 0 if x <= 4 else 1 if 4 < x <= 6 else 2 if x > 6 else x)

    print(df_val_1['日剂量分组'].unique())

    df_val_1['预测准确率'] = (df_val_1['日剂量预测值'] - df_val_1['日剂量']) / df_val_1['日剂量']
    df_val_1['预测准确率'] = df_val_1['预测准确率'].apply(lambda x: 1 if abs(x) <= 0.2 else 0)

    aaa = df_val_1[df_val_1['日剂量分组'] == 0]
    bbb = df_val_1[df_val_1['日剂量分组'] == 1]
    ccc = df_val_1[df_val_1['日剂量分组'] == 2]

    print(aaa.shape[0])
    print(bbb.shape[0])
    print(ccc.shape[0])

    print(aaa[aaa['预测准确率'] == 1].shape[0] / aaa.shape[0])
    print(bbb[bbb['预测准确率'] == 1].shape[0] / bbb.shape[0])
    print(ccc[ccc['预测准确率'] == 1].shape[0] / ccc.shape[0])

    from pylab import mpl

    mpl.rcParams['font.sans-serif'] = ['SimHei']  ##绘图显示中文
    mpl.rcParams['axes.unicode_minus'] = False
    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib import rc

    rc('mathtext', default='regular')

    sub_axix = filter(lambda x: x % 200 == 0, list(range(df_val_1.shape[0])))
    # plt.title('测试集他克莫司日剂量真实值和预测值对比折线图')
    plt.plot(list(range(df_val_1.shape[0])), np.array(list(df_val_1['日剂量预测值'])),
             color=(0.32941176470588235, 0.7294117647058823, 0.7490196078431373),
             label='The Predictive Value of Tac Daily Dose(mg)')
    # plt.plot(sub_axix, test_acys, color='red', label='testing accuracy')
    plt.plot(list(range(df_val_1.shape[0])), np.array(list(df_val_1['日剂量'])), color='indianred',
             label='The True Value of Tac Daily Dose(mg)')
    # plt.plot(list(range(test_x_1.shape[0])), thresholds, color='blue', label='threshold')
    plt.legend(bbox_to_anchor=(1.1, 1))  # 显示图例

    plt.xlabel('Number of Events(unit)')
    plt.ylabel('Tac Daily Dose(mg)')
    # plt.show()
    plt.savefig(jpg_path + "/验证集折线图.jpg", dpi=300)


    print('---------------------验证模型-首次检测达标和部分缺基因型脱敏----------------------------------')
    df_val_1 = pd.read_excel(project_path + '/data/test_dataset/他克莫司验证模型20190730 首次检测达标.xlsx')
    df_val_2 = pd.read_excel(project_path + '/data/test_dataset/他克莫司验证模型20190717部分缺基因型脱敏.xlsx')

    print(df_val_1.columns)
    print(df_val_2.columns)

    df_val_1 = df_val_1.rename(columns={'患者ID': '病人ID', '移植手术时间': '肾移植手术时间', 'FK506首次用药日剂量': '日剂量', \
                                        '移植后3个月内首次服用FK506后首次检测TDM值（8-15 ng/ml)': '本次他克莫司血药浓度（在8～15 ng/ml范围内）'})
    df_val_2 = df_val_2.rename(columns={'年龄(岁）': '年龄', '身高(cm)': '身高', '体重(kg)': '体重', '上一次监测他克莫司血药浓度值(ng/ml)': '上一次TDM值', \
                                        ' CYP3A5*3基因表型': 'CYP3A5*3', '医嘱他克莫司的用药日剂量（mg）': '日剂量',
                                        '上一次他克莫司医嘱日剂量(mg)': '上一次日剂量'})

    print(df_val_1.shape)
    print(df_val_2.shape)

    df_val_1['是否首次用药'] = 1
    df_val_2['是否首次用药'] = 0
    df_val = pd.concat([df_val_1, df_val_2], axis=0)

    print(df_val.shape)
    print(len(df_val['病人ID'].unique()))
    print(df_val.columns)

    df_val = df_val[['病人ID', '是否首次用药', '年龄', '身高', '体重', '肾移植手术时间', '上一次TDM值', '上一次日剂量', 'CYP3A5*3', '红细胞压积',
                     '本次他克莫司血药浓度（在8～15 ng/ml范围内）', '日剂量']]

    df_val['移植术后天数'] = np.nan
    df_val['糖皮质激素日剂量'] = np.nan
    df_val['MPA类药物'] = np.nan

    print(df_val.columns)

    for i in df_val.columns:
        if df_val[i].isnull().sum() > 0:
            print(i)

    ## 按病人ID分组，年龄，身高、肾移植时间、'CYP3A5*3'
    aaa = []
    for i in np.unique(df_val['病人ID']):
        temp = df_val[df_val['病人ID'] == i]
        temp = temp.reset_index()
        del temp['index']

        ## 补充基因数据
        if temp['CYP3A5*3'].isnull().sum() < temp.shape[0]:
            temp['CYP3A5*3'] = temp[temp['CYP3A5*3'].notnull()].reset_index().loc[0, 'CYP3A5*3']
        aaa.append(temp)

    df_val_new = aaa[0]
    for i in range(1, len(aaa)):
        df_val_new = pd.concat([df_val_new, aaa[i]], axis=0)  # 将groupby后的list类型转换成DataFrame
    df_val_new = df_val_new.reset_index()
    del df_val_new['index']


    writer = pd.ExcelWriter(project_path + '/data/test_dataset/df_验证集变量待补充.xlsx')
    df_val_new.to_excel(writer)
    writer.save()

    print(df_val_new)  ##2434457

if __name__=='__main__':
    sys.exit(main())