# _*_ coding: utf-8 _*_
# @Time: 2021/9/3015:39 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description: 他克莫司DM项目学习：2.2 调整方案数据处理

import os
import sys

import pandas as pd
import numpy as np
import pymysql as MySQLDB

import re
from datetime import datetime
from pandas import Series, DataFrame

project_path = os.getcwd()

# 日期转成数值
def diff(x):
    try:
        temp = int(x.split('-')[0])*365+int(x.split('-')[1])*30+int(x.split('-')[2])
        return temp
    except:
        return np.nan

# 数值转成日期
def date(x):
    if (x%365) < 30:
        year = int(x/365) - 1
    else:
        year = int(x/365)
    if (x-year*365)%30==0:
        month = int((x-year*365)/30) - 1
    else:
        month = int((x-year*365)/30)
    day = int(x - year*365 - month*30)
    return str(year) + '-' + str(month) + '-' + str(day)

def judge_float(x):
    try:
        a=float(x)
        return a
    except :
        return np.NaN
    
drug_record_tcms = pd.read_excel(project_path + '/data/data_from_mysql/df_他克莫司用药数据.xlsx')
drug_record_other = pd.read_excel(project_path + '/data/data_from_mysql/df_其他用药数据.xlsx')
surgical_record = pd.read_excel(project_path + '/data/data_from_mysql/df_肾移植手术时间.xlsx')
diagnose_record = pd.read_excel(project_path + '/data/data_from_mysql/df_诊断数据.xlsx')
gene_record = pd.read_excel(project_path + '/data/data_from_mysql/他克莫司用药模型建立-脱敏有基因型 20190627.xlsx')
test_index = pd.read_excel(project_path + '/data/data_from_mysql/df_检验序号索引.xlsx')

# 连接数据库
conn = MySQLDB.connect(host='localhost', port=3306, user='root', password='123456', db='mdnov_ciip_ipharma_zdy', charset='UTF8')
cursor = conn.cursor()
cursor.execute("Select version()")
for i in cursor:
    print(i)
    
#读取数据
try:
    sql = 'select TEST_RECORD_ID,PROJECT_CODE,PROJECT_NAME,TEST_RESULT,RESULT_UNIT,IS_NORMAL,REFER_SCOPE from test_result;'
    test_result = pd.read_sql(sql, conn)
except MySQLDB.err.ProgrammingError as e:
    print('test_result Error is ' + str(e))
    sys.exit()

test_index_result=pd.merge(test_index,test_result,on=['TEST_RECORD_ID'],how='inner')
test_index_result=test_index_result.reset_index()
del test_index_result['index']

# 检验结果按照病人id排序
test_index_result=test_index_result.sort_values(['PATIENT_ID'],ascending=1)

print(len(np.unique(drug_record_tcms['PATIENT_ID'])))
print(len(np.unique(drug_record_other['PATIENT_ID'])))
print(len(np.unique(surgical_record['PATIENT_ID'])))
print(len(np.unique(test_index_result['PATIENT_ID'])))
print(len(np.unique(diagnose_record['PATIENT_ID'])))
print(len(np.unique(gene_record['病人ID'])))

print(test_index_result.shape)

# 读取调整方案符合终点事件
Tacrolimus_tdm_adjust = pd.read_excel(project_path + '/data/data_from_mysql/df_调整方案符合终点事件.xlsx')
Tacrolimus_tdm_adjust=Tacrolimus_tdm_adjust.drop(['Unnamed: 0'],axis=1)
print(Tacrolimus_tdm_adjust.shape)

#  调整方案所有变量数据清洗
print('-------------------------调整方案所有变量数据清洗---------------------------')
print(Tacrolimus_tdm_adjust.columns)

# 处理年龄、性别
gene_record['年龄']=gene_record['手术时间'].astype('str').apply(lambda x: int(x.split('-')[0]))-gene_record['年龄'].astype('str').apply(lambda x: int(x.split('-')[0]))
gene_record=gene_record.rename(columns={'性别1=12=2':'性别','病人ID':'PATIENT_ID','GYP3A5':'CYP3A5'})
gene_record['性别']=gene_record['性别'].apply(lambda x: 0 if x==1 else 1 if x==2 else x)

## 计算BMI
gene_record['BMI']=(gene_record['体重']*10000)/(gene_record['身高']*gene_record['身高'])

Tacrolimus_tdm_adjust=Tacrolimus_tdm_adjust.rename(columns={'TEST_RESULT':'TDM检测结果'})
tcms_adjust_gene=pd.merge(Tacrolimus_tdm_adjust[['PATIENT_ID','日剂量', 'FREQUENCY','TDM检测结果','上一次TDM值','上一次日剂量','移植术后天数','是否亲属同体']],\
                               gene_record[['PATIENT_ID', '性别','年龄','身高', '体重','BMI','CYP3A5','ABCB1']],\
                               on=['PATIENT_ID'],how='left')
tcms_adjust_gene=tcms_adjust_gene.reset_index()
del tcms_adjust_gene['index']

print(np.unique(tcms_adjust_gene[tcms_adjust_gene['年龄'].isnull()]['PATIENT_ID']))

# 基因编码0，1，2，3
tcms_adjust_gene['CYP3A5'] = tcms_adjust_gene['CYP3A5'].apply(
    lambda x: 0 if x == 'AA' else 1 if (x == 'AG') | (x == 'GA') else 2 if x == 'GG' else x)
tcms_adjust_gene['ABCB1'] = tcms_adjust_gene['ABCB1'].apply(
    lambda x: 0 if x == 'CC' else 1 if (x == 'CT') | (x == 'TC') else 2 if x == 'TT' else x)

print(tcms_adjust_gene.shape)
print(tcms_adjust_gene.head())


## 其他用药信息提取（酶抑制剂、康唑类、钙通道阻滞剂、拉唑类、酶诱导剂、糖皮质激素、MPA类药物）
print('------------------------其他用药信息提取----------------------------')
print(drug_record_other.columns)
print(Tacrolimus_tdm_adjust.columns)

## 只提取调整方案中的病人信息
tcms_adjust_other = pd.merge(
    Tacrolimus_tdm_adjust.drop_duplicates(subset=['PATIENT_ID'], keep='first')[['PATIENT_ID']], \
    drug_record_other, on=['PATIENT_ID'], how='left')

print(tcms_adjust_other.shape)
print(len(np.unique(tcms_adjust_other['PATIENT_ID'])))
print(tcms_adjust_other.columns)

import datetime

# 字符串转换为日期
def str_to_datatime(x):
    try:
        a = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        return a
    except:
        return np.NaN

for i in range(Tacrolimus_tdm_adjust.shape[0]):
    ## 从其他用药数据集中提取该病人的所有其它用药，从中筛选出符合联合用药时间段的用药信息,依次判断该时段内是否使用规定药物，如果使用则标记1
    ## 该patient_id的其他用药记录
    temp = tcms_adjust_other[tcms_adjust_other['PATIENT_ID'] == Tacrolimus_tdm_adjust.loc[i, 'PATIENT_ID']]
    temp['END_DATETIME'] = temp['END_DATETIME'].astype('str').apply(str_to_datatime)
    temp_1 = temp[(temp['END_DATETIME'] >= Tacrolimus_tdm_adjust.loc[i, '服药开始时间'] - datetime.timedelta(days=3)) & \
                  ((temp['START_DATETIME'] <= Tacrolimus_tdm_adjust.loc[i, 'TDM检测时间']))]

    temp_1['康唑类'] = temp_1['DRUG_NAME'].astype('str').apply(lambda x: 1 if '康唑' in x else 0)
    if temp_1[temp_1['康唑类'] == 1].shape[0] > 0:
        Tacrolimus_tdm_adjust.loc[i, '康唑类'] = 1
    else:
        Tacrolimus_tdm_adjust.loc[i, '康唑类'] = 0

    temp_1['钙通道阻滞剂'] = temp_1['DRUG_NAME'].astype('str').apply(lambda x: 1 if ('地尔硫卓' in x) | ('维拉帕米' in x) | ('地平' in x) else 0)
    if temp_1[temp_1['钙通道阻滞剂'] == 1].shape[0] > 0:
        Tacrolimus_tdm_adjust.loc[i, '钙通道阻滞剂'] = 1
    else:
        Tacrolimus_tdm_adjust.loc[i, '钙通道阻滞剂'] = 0

    temp_1['拉唑类'] = temp_1['DRUG_NAME'].astype('str').apply(lambda x: 1 if ('奥美拉唑' in x) | ('泮托拉唑' in x) else 0)
    if temp_1[temp_1['拉唑类'] == 1].shape[0] > 0:
        Tacrolimus_tdm_adjust.loc[i, '拉唑类'] = 1
    else:
        Tacrolimus_tdm_adjust.loc[i, '拉唑类'] = 0

    temp_1['酶诱导剂'] = temp_1['DRUG_NAME'].astype('str').apply(lambda x: 1 if ('苯妥英钠' in x) | ('巴比妥' in x) | ('利福平' in x) | 
                                                                            ('卡马西平' in x) | ('圣约翰草' in x) | ('贯叶连翘' in x) else 0)
    if temp_1[temp_1['酶诱导剂'] == 1].shape[0] > 0:
        Tacrolimus_tdm_adjust.loc[i, '酶诱导剂'] = 1
    else:
        Tacrolimus_tdm_adjust.loc[i, '酶诱导剂'] = 0

    temp_1['MPA类药物'] = temp_1['DRUG_NAME'].astype('str').apply(lambda x: 1 if ('吗替麦考酚酯' in x) | ('霉酚酸' in x) else 0)
    if temp_1[temp_1['MPA类药物'] == 1].shape[0] > 0:
        Tacrolimus_tdm_adjust.loc[i, 'MPA类药物'] = 1
    else:
        Tacrolimus_tdm_adjust.loc[i, 'MPA类药物'] = 0

    wuzhi = temp_1[temp_1['DRUG_NAME'].str.contains('五酯软胶囊')]
    wuzhi['五酯软胶囊次剂量'] = wuzhi['DOSAGE'].astype('str').apply(lambda x: '0' + x if x[0] == '.' else x)
    wuzhi['五酯软胶囊次剂量'] = wuzhi['五酯软胶囊次剂量'].astype('str').apply(
        lambda x: float(x.replace('mg', '')) / 1000 if 'm' in x else float(x.replace('g', '')))
    wuzhi = wuzhi[wuzhi['FREQUENCY'] != 'ACM早餐前']
    one = ['PRN必要时用', 'QD每天', 'QD每天一次']
    two = ['BID719每日两次(7:00-19:00)', 'BIDCXY每日两次(6,16)', 'BID一日两次', 'BID日两次', 'BID每日两次', 'BOD间隔2日，每日两次', \
           'Q12H12间隔12小时(0,12)', 'Q12H间隔12小时', 'Q.M.每天早上服', 'QN每晚一次']
    three = ['Q8H间隔8小时', 'TID3每日三次(8,16,20)', 'TID一日三次', 'TID日三次', 'TID每日三次']
    wuzhi['FREQUENCY'] = wuzhi['FREQUENCY'].apply(
        lambda x: 1 if x in one else 2 if x in two else 3 if x in three else x)
    ## 计算日剂量
    wuzhi['五酯软胶囊日剂量'] = wuzhi['五酯软胶囊次剂量'] * wuzhi['FREQUENCY']
    ## 每个病人id只保留第一次使用五酯软胶囊的日剂量
    wuzhi = wuzhi.reset_index()
    del wuzhi['index']
    if wuzhi.shape[0] > 0:
        Tacrolimus_tdm_adjust.loc[i, '五酯软胶囊日剂量'] = wuzhi.loc[0, '五酯软胶囊日剂量']
    else:
        Tacrolimus_tdm_adjust.loc[i, '五酯软胶囊日剂量'] = 0

    jisu = temp_1[(temp_1['DRUG_NAME'].str.contains('甲泼尼龙')) | (temp_1['DRUG_NAME'].str.contains('泼尼松'))]
    jisu['糖皮质激素次剂量'] = jisu['DOSAGE'].astype('str').apply(lambda x: '0' + x if x[0] == '.' else x)
    jisu['糖皮质激素次剂量'] = jisu['糖皮质激素次剂量'].astype('str').apply(
        lambda x: float(x.replace('g', '')) * 1000 if 'm' not in x else float(x.replace('mg', '')))
    ## 将泼尼松的剂量转换成甲泼尼龙的等价剂量，泼尼松剂量*1.25
    jisu = jisu.reset_index()
    del jisu['index']
    for j in range(jisu.shape[0]):
        if '泼尼松' in str(jisu.loc[j, 'DRUG_NAME']):
            jisu.loc[j, '糖皮质激素次剂量'] = jisu.loc[j, '糖皮质激素次剂量'] * 1.25
    ## 将频次修改为数字
    jisu = jisu[jisu['FREQUENCY'] != 'ACM早餐前']
    jisu = jisu[jisu['FREQUENCY'] != 'PCM早餐后']
    jisu = jisu[jisu['FREQUENCY'] != 'QD(06)每天一次(6点)']
    jisu = jisu[jisu['FREQUENCY'] != 'QD(07)每天一次(7点)']
    jisu = jisu[jisu['FREQUENCY'] != 'QD(11)每天一次(11点)']
    jisu = jisu[jisu['FREQUENCY'] != 'QD(16)每天一次(16点)']
    jisu = jisu[jisu['FREQUENCY'] != 'QD(17)每天一次(17点)']
    one = ['QD每天', 'QD每天一次', 'PRN必要时用', 'ST立即用', 'PRN必要时', 'QOD隔天一次']
    two = ['BID每日两次', 'Q.M.每天早上服', 'QN每晚一次', 'BID8每日两次(口服)', 'BID一日两次', 'BID日两次', 'Q12H间隔12小时']
    three = ['TH.顿服', 'TID每日三次']
    four = ['Q3H8间隔3小时(8,11,14,17)']
    jisu['FREQUENCY'] = jisu['FREQUENCY'].apply(
        lambda x: 1 if x in one else 2 if x in two else 3 if x in three else 4 if x in four else x)
    ## 计算激素日剂量
    jisu['糖皮质激素日剂量'] = jisu['糖皮质激素次剂量'] * jisu['FREQUENCY']
    ## 每个病人id只保留第一次使用激素的日剂量
    jisu = jisu.reset_index()
    del jisu['index']
    if jisu.shape[0] > 0:
        Tacrolimus_tdm_adjust.loc[i, '糖皮质激素日剂量'] = jisu.loc[0, '糖皮质激素日剂量']
    else:
        Tacrolimus_tdm_adjust.loc[i, '糖皮质激素日剂量'] = 0

print(Tacrolimus_tdm_adjust)



## 看用药里面有没有全是0的变量
col_var = ['五酯软胶囊日剂量', '康唑类', '钙通道阻滞剂', '拉唑类', '酶诱导剂', 'MPA类药物', '糖皮质激素日剂量']
for i in col_var:
    Tacrolimus_tdm_adjust[i] = Tacrolimus_tdm_adjust[i].replace(np.nan, 0)
    print(Tacrolimus_tdm_adjust[Tacrolimus_tdm_adjust[i] == 1].shape[0])
    print(Tacrolimus_tdm_adjust[i].isnull().sum())



## 将用药信息合并到调整方案建模数据集中
tcms_adjust_gene_1 = pd.concat([tcms_adjust_gene,Tacrolimus_tdm_adjust[
    ['五酯软胶囊日剂量', '康唑类', '钙通道阻滞剂', '拉唑类', '酶诱导剂', 'MPA类药物', '糖皮质激素日剂量']]], axis=1)

print(tcms_adjust_gene_1.shape)
print(tcms_adjust_gene_1.head())

## 提取检验信息
print('---------------------------提取检验信息------------------------------------')
print(test_index_result.columns)
print(test_index_result.shape)

## 只提取调整方案中的病人信息
tcms_adjust_test_result = pd.merge(Tacrolimus_tdm_adjust.drop_duplicates(subset=['PATIENT_ID'], keep='first')[['PATIENT_ID']], \
                         test_index_result, on=['PATIENT_ID'], how='left')

print(tcms_adjust_test_result.shape)
print(len(np.unique(tcms_adjust_test_result['PATIENT_ID'])))

tcms_adjust_test_result['PROJECT_NAME'] = tcms_adjust_test_result['PROJECT_NAME'].astype('str')

print(tcms_adjust_test_result[tcms_adjust_test_result['PROJECT_NAME'].str.contains('红细胞')]['PROJECT_NAME'].unique())
print(tcms_adjust_test_result[tcms_adjust_test_result['PROJECT_NAME'].str.contains('血红蛋白')]['PROJECT_NAME'].unique())



## 肾功能*（尿酸、肌酐）、肝功能（谷草、总胆红素）、血常规（红细胞压积）、中性粒细胞%（或中性粒细胞百分数）、淋巴细胞
tcms_adjust_test_result['PROJECT_NAME'] = tcms_adjust_test_result['PROJECT_NAME'].replace('中性粒细胞%', '中性粒细胞百分数')
tcms_adjust_test_result['PROJECT_NAME'] = tcms_adjust_test_result['PROJECT_NAME'].replace('中性粒细胞百分数 ', '中性粒细胞百分数')
tcms_adjust_test_result['PROJECT_NAME'] = tcms_adjust_test_result['PROJECT_NAME'].replace('淋巴细胞%', '淋巴细胞百分数')
tcms_adjust_test_result['PROJECT_NAME'] = tcms_adjust_test_result['PROJECT_NAME'].replace('红细胞', '红细胞计数')
tcms_adjust_test_result['PROJECT_NAME'] = tcms_adjust_test_result['PROJECT_NAME'].replace('红细胞数', '红细胞计数')
tcms_adjust_test_result['PROJECT_NAME'] = tcms_adjust_test_result['PROJECT_NAME'].replace('红细胞计数 ', '红细胞计数')
tcms_adjust_test_result_2 = tcms_adjust_test_result[(tcms_adjust_test_result['PROJECT_NAME'] == '尿酸') | (tcms_adjust_test_result['PROJECT_NAME'] == '肌酐') | \
                              (tcms_adjust_test_result['PROJECT_NAME'] == '谷草转氨酶') | (tcms_adjust_test_result['PROJECT_NAME'] == '总胆红素') | \
                              (tcms_adjust_test_result['PROJECT_NAME'] == '红细胞压积') | (tcms_adjust_test_result['PROJECT_NAME'] == '红细胞计数') | \
                              (tcms_adjust_test_result['PROJECT_NAME'] == '血红蛋白') | \
                              (tcms_adjust_test_result['PROJECT_NAME'] == '中性粒细胞百分数') | (tcms_adjust_test_result['PROJECT_NAME'] == '淋巴细胞百分数')]

tcms_adjust_test_result_2 = tcms_adjust_test_result_2.reset_index()
del tcms_adjust_test_result_2['index']

print(tcms_adjust_test_result_2.shape)
print(np.unique(tcms_adjust_test_result_2['PROJECT_NAME']))


for i in range(Tacrolimus_tdm_adjust.shape[0]):
    ## 从检验数据集中提取该病人的所有检验，从中筛选出符合时间段的检验信息，取离服药时间最近的一次检验
    temp = tcms_adjust_test_result_2[tcms_adjust_test_result_2['PATIENT_ID'] == Tacrolimus_tdm_adjust.loc[i, 'PATIENT_ID']]
    # temp['COLLECT_TIME']=temp['COLLECT_TIME'].astype('str').apply(str_to_datatime)
    temp_1 = temp[(temp['COLLECT_TIME'] >= Tacrolimus_tdm_adjust.loc[i, '服药开始时间'] - datetime.timedelta(days=10)) & \
                  ((temp['COLLECT_TIME'] <= Tacrolimus_tdm_adjust.loc[i, '服药开始时间']))]

    ## 同一个病人ID的相同检验项，取最接近服药的那一次检验
    temp_1 = temp_1.drop_duplicates(subset=['PATIENT_ID', 'PROJECT_NAME'], keep='last')
    temp_1 = temp_1.reset_index()
    del temp_1['index']

    ## 检验项转置为表头
    if temp_1.shape[0] > 0:
        temp_1['TEST_RESULT'] = temp_1['TEST_RESULT'].apply(judge_float)
        temp_2 = temp_1.pivot_table('TEST_RESULT', ['PATIENT_ID'], 'PROJECT_NAME')
        temp_2 = temp_2.reset_index()
        for j in temp_2.columns[1:]:
            Tacrolimus_tdm_adjust.loc[i, j] = temp_2.loc[0, j]

print(Tacrolimus_tdm_adjust)



for i in np.unique(tcms_adjust_test_result_2['PROJECT_NAME']):
    print(Tacrolimus_tdm_adjust[i].notnull().sum())
    print(Tacrolimus_tdm_adjust[i].isnull().sum())



## 将检测数据并到调整方案建模数据集中
print('------------------------------将检测数据合并到调整方案建模数据集中----------------------------')
tcms_adjust_gene_2 = pd.concat([tcms_adjust_gene_1, Tacrolimus_tdm_adjust[
                    ['中性粒细胞百分数', '尿酸', '总胆红素', '淋巴细胞百分数', '红细胞压积', '红细胞计数', '肌酐', '血红蛋白','谷草转氨酶']]], axis=1)

print(tcms_adjust_gene_2.shape)
print(tcms_adjust_gene_2.columns)


## 诊断信息提取（生理病理状况（腹泻、腹痛。。。）、高血压、糖尿病、幽门螺旋杆菌感染）
print('--------------------------------------诊断信息提取-------------------------------------------')
print(diagnose_record.columns)

diagnose_record['高血压'] = diagnose_record['DIAGNOSTIC_CONTENT'].apply(lambda x: 1 if '高血压' in x else 0)
diagnose_record['糖尿病'] = diagnose_record['DIAGNOSTIC_CONTENT'].apply(lambda x: 1 if '糖尿病' in x else 0)
diagnose_record['生理病理状况'] = diagnose_record['DIAGNOSTIC_CONTENT'].apply(
    lambda x: 1 if ('腹泻' in x) | ('腹痛' in x) | ('腹胀' in x) | ('嗳气' in x) | ('纳差' in x) | ('胃肠动力恢复' in x) else 0)
# diagnose_record['幽门螺旋杆菌感染']=diagnose_record['DIAGNOSTIC_CONTENT'].apply(lambda x: 1 if '幽门螺旋杆菌' in x else 0)

print(diagnose_record[diagnose_record['高血压'] == 1].shape[0])
print(diagnose_record.shape)

## 将诊断信息合并到调整方案建模数据集中
print('-----------------------将诊断信息合并到调整方案建模数据集中------------------------------')
tcms_adjust_gene_3 = pd.merge(tcms_adjust_gene_2, diagnose_record[['PATIENT_ID', '高血压', '糖尿病', '生理病理状况']],
                                   on=['PATIENT_ID'], how='left')
tcms_adjust_gene_3 = tcms_adjust_gene_3.rename(columns={'是否亲属同体': '是否亲属活体'})

print(tcms_adjust_gene_3.shape)
print(tcms_adjust_gene_3.columns)
print(np.unique(tcms_adjust_gene_3['ABCB1'].astype('str')))
print(tcms_adjust_gene_3.head())

writer = pd.ExcelWriter(project_path + '/data/data_from_mysql/df_调整方案建模数据集.xlsx')
tcms_adjust_gene_3.to_excel(writer)
writer.save()