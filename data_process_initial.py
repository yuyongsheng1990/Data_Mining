# _*_ coding: utf-8 _*_
# @Time: 2021/9/3015:39 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description: 他克莫司DM项目学习：2.1 初始方案数据处理

import pymysql as MySQLDB
import pandas as pd
import numpy as np

import re
import sys
import os

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

# 判断数据是否是浮点型
def judge_float(x):
    try:
        a=float(x)
        return a
    except :
        return np.NaN

# 读取保存的excel数据
print('------------------------读取excel数据--------------------------------------')
surgical_record = pd.read_excel(project_path + '/data/data_from_mysql/df_肾移植手术时间.xlsx')
drug_record_tcms = pd.read_excel(project_path + '/data/data_from_mysql/df_他克莫司用药数据.xlsx')
drug_record_other = pd.read_excel(project_path + '/data/data_from_mysql/df_其他用药数据.xlsx')
test_record_index = pd.read_excel(project_path + '/data/data_from_mysql/df_检验序号索引.xlsx')
diagnose_record = pd.read_excel(project_path + '/data/data_from_mysql/df_诊断数据.xlsx')
base_gene_record = pd.read_excel(project_path + '/data/data_from_mysql/他克莫司用药模型建立-脱敏有基因型 20190627.xlsx')
# print(drug_record_tcms)

surgical_record=surgical_record.drop(['Unnamed: 0'],axis=1)
drug_record_tcms=drug_record_tcms.drop(['Unnamed: 0'],axis=1)
drug_record_other=drug_record_other.drop(['Unnamed: 0'],axis=1)
test_record_index=test_record_index.drop(['Unnamed: 0'],axis=1)
diagnose_record=diagnose_record.drop(['Unnamed: 0'],axis=1)
# print(drug_record_tcms)

# 连接mysql数据库
conn = MySQLDB.connect(host='localhost', port=3306, user='root', password='123456', db='mdnov_ciip_ipharma_zdy', charset='UTF8')
cursor = conn.cursor()
cursor.execute("Select version()")
for i in cursor:
    print(i)


#读取检测结果数据
print('-----------------------从mysql读取test_result数据----------------------------------')
try:
    sql = 'select TEST_RECORD_ID,PROJECT_CODE,PROJECT_NAME,TEST_RESULT,RESULT_UNIT,IS_NORMAL,REFER_SCOPE from test_result;'
    test_result = pd.read_sql(sql, conn)
except MySQLDB.err.ProgrammingError as e:
    print('test_result Error is ' + str(e))
    sys.exit()


test_record_result =pd.merge(test_record_index,test_result,on=['TEST_RECORD_ID'],how='inner')
test_record_result = test_record_result.reset_index()
del test_record_result['index']

# 检验结果按照病人id排序
test_record_result=test_record_result.sort_values(['PATIENT_ID'],ascending=1)
print(test_record_result.shape)

print(len(np.unique(drug_record_tcms['PATIENT_ID'])))
print(len(np.unique(drug_record_other['PATIENT_ID'])))
print(len(np.unique(surgical_record['PATIENT_ID'])))
print(len(np.unique(test_record_result['PATIENT_ID'])))
print(len(np.unique(diagnose_record['PATIENT_ID'])))
print(len(np.unique(base_gene_record['病人ID'])))

print(test_record_result.shape)

# 从检验中提取他克莫司tdm检测结果
print('----------------------从检验中提取他克莫司tdm检测结果--------------------------------')
# test_record_result=test_record_result[test_record_result['PROJECT_NAME'].notnull()]
# print(test_record_result.shape)
test_record_result_tdm=test_record_result[test_record_result['PROJECT_NAME']=='他克莫司']
test_record_result_tdm=test_record_result_tdm.reset_index()
del test_record_result_tdm['index']
print(test_record_result_tdm.shape)

print(drug_record_tcms.columns)
# print(len(drug_record_tcms.columns))
print(test_record_result_tdm.columns)
# print(len(test_record_result_tdm.columns))


# 合并他克莫司用药和tdm检测信息，将同一个病人的用药和tdm检测按时间排序，形成用药-tdm的时间序列
print('----------------------合并他克莫司用药和tdm检测信息----------------------------------')
tcms_tdm=pd.concat([drug_record_tcms.rename(columns={'START_DATETIME':'time'}),test_record_result_tdm.rename(columns={'COLLECT_TIME':'time'})],axis=0)
print(tcms_tdm.columns)
# print(len(tcms_tdm.columns))
tcms_tdm = tcms_tdm[['PATIENT_ID','DRUG_NAME','DOSAGE','FREQUENCY','PROJECT_NAME','TEST_RESULT','time','END_DATETIME']]
print(tcms_tdm.shape)
print(tcms_tdm.head())

aaa=[]
for i in np.unique(tcms_tdm['PATIENT_ID']):
    temp=tcms_tdm[tcms_tdm['PATIENT_ID']==i]
    temp=temp.sort_values(['time'],ascending=1)
    aaa.append(temp)

# list类型数据转换成DataFrame
tcms_tdm_1 = aaa[0]
for i in range(1,len(aaa)):
    tcms_tdm_1=pd.concat([tcms_tdm_1,aaa[i]],axis=0) #将groupby后的list类型转换成DataFrame
tcms_tdm_1=tcms_tdm_1.reset_index()
del tcms_tdm_1['index']
# print(tcms_tdm_1.shape)

# 将完全相同的记录去重
tcms_tdm_1=tcms_tdm_1.drop_duplicates(subset=tcms_tdm_1.columns,keep='last')
tcms_tdm_1=tcms_tdm_1.reset_index()
del tcms_tdm_1['index']

print(tcms_tdm_1.shape)
print(tcms_tdm_1.head())
print(tcms_tdm_1.columns)

# 提取肾移植手术之后的他克莫司用药信息
print('----------------------提取肾移植手术之后的他克莫司用药信息----------------------------------')
# 向他克莫司和tdm检测表中加入手术信息
tcms_tdm_surgery = pd.merge(tcms_tdm_1,surgical_record,on=['PATIENT_ID'],how='inner')

print(tcms_tdm_surgery.shape)
print(tcms_tdm_surgery.columns)

# 截取肾移植手术后3个月内的他克莫司用药信息
import datetime
tcms_tdm_surgery=tcms_tdm_surgery[(tcms_tdm_surgery['time']>=tcms_tdm_surgery['SURGERY_DATE']-datetime.timedelta(days=2))&(tcms_tdm_surgery['time']<=tcms_tdm_surgery['SURGERY_DATE']+datetime.timedelta(days=90))]
tcms_tdm_surgery=tcms_tdm_surgery.reset_index()
del tcms_tdm_surgery['index']

# print(tcms_tdm_surgery.shape)
# print(tcms_tdm_surgery.loc[:, ['PATIENT_ID', 'time']])
print(len(np.unique(tcms_tdm_surgery['PATIENT_ID'])))


# 删除用药信息不完整及没有 TDM 检测信息的病人 id
print('------------------删除用药信息不完整及没有TDM检测信息的病人id----------------------------------')
# tcms用药数据，tdm检测数据不同行，时间升序排列
# 按病人id分组，有用药信息和tdm检测，否则删除该样本。
for i in np.unique(tcms_tdm_surgery['PATIENT_ID']):
    temp = tcms_tdm_surgery[tcms_tdm_surgery['PATIENT_ID']==i]
    temp = temp.reset_index()
    del temp['index']
    trigger = True
    for j in range(temp.shape[0]):
        # 判断是否有用药数据
        if str(temp.loc[j,'DRUG_NAME']) != 'nan':
            # 判断是否有tdm检测，如果都有则保留，否则就删除该样本
            for k in range(j+1, temp.shape[0]):
                if str(temp.loc[k,'PROJECT_NAME']) != 'nan':
                    trigger = False
                    break
            break

    if trigger:
        print(i)
        # print(temp)
        tcms_tdm_surgery = tcms_tdm_surgery[tcms_tdm_surgery['PATIENT_ID'] != i]

print(tcms_tdm_surgery.shape)
print(len(np.unique(tcms_tdm_surgery['PATIENT_ID'])))
    # 删除上述第一条不是用药信息的病人id
#     但是，如果为了删除不完整用药信息的话，应该只删除限定时间内第一次tcms用药之前的tdm检测记录；如果是为了避免之前tcms用药影响限定时间内的tcms用药效果，
#     我们又如何保证当前tcms用药是第一次tcms用药，所以之前的删除不合理！

'''
# 删除没有tdm检测的病人id
for i in np.unique(tcms_tdm_surgery['PATIENT_ID']):
    temp = tcms_tdm_surgery[tcms_tdm_surgery['PATIENT_ID'] == i]
    temp = temp.reset_index()
    del temp['index']

    if temp['PROJECT_NAME'].isnull().sum() == temp.shape[0]:
        print(i)
        tcms_tdm_surgery = tcms_tdm_surgery[tcms_tdm_surgery['PATIENT_ID'] != i]
'''


# 把剂量转换成数值
tcms_tdm_surgery['DOSAGE']=tcms_tdm_surgery['DOSAGE'].astype('str').apply(lambda x: '0.5mg' if x=='.5mg' else x)
tcms_tdm_surgery['DOSAGE']=tcms_tdm_surgery['DOSAGE'].apply(lambda x: x.replace('mg',''))
tcms_tdm_surgery['DOSAGE']=tcms_tdm_surgery['DOSAGE'].apply(lambda x: np.nan if x=='nan' else float(x))

print(tcms_tdm_surgery.columns)
print(np.unique(tcms_tdm_surgery['FREQUENCY'].astype('str')))


## 多条用药保留最后一次，如果是那几种特殊的需要加和的，进行加和。
## 将所有频次为QN每晚一次和Q.M.每天早上服合并为一条，日剂量=早+晚，单次剂量为日剂量/2，频次为BID日两次(自定义)，这是同一天之内而言的医嘱多次用药的，每天早上几点一次+每晚几点一次，不是不同天的，所以需要做一个用药频次的规范化处理。
## 如果有多次用药，只保留最后一条用药记录（允许有连续多条tdm检测）
print('------------------多次用药信息dosage处理----------------------------------')
col_sum = ['12PM中午12点服', 'Q.M.每天早上服', 'QD(06)每天一次(6点)', 'QD(16)每天一次(16点)', 'QD(17)每天一次(17点)', 'QN(23)每晚(23:00)',
           'QN每晚1次', 'QN每晚一次']
all_id = []
for i in np.unique(tcms_tdm_surgery['PATIENT_ID']):
    temp = tcms_tdm_surgery[tcms_tdm_surgery['PATIENT_ID'] == i]
    temp = temp.reset_index()
    del temp['index']
    # print(temp[['PATIENT_ID', 'DRUG_NAME','PROJECT_NAME','time']])

    ## Tacrolimus_tdm=Tacrolimus_tdm[['PATIENT_ID','DRUG_NAME','DOSAGE','FREQUENCY','PROJECT_NAME','TEST_RESULT','time','END_DATETIME']]
    between_id = []
    for j in reversed(range(temp.shape[0])):
        ## 如果本行是检验，保留
        if str(temp.loc[j, 'PROJECT_NAME']) == '他克莫司':
            between_id.append(temp.iloc[[j]])
            ## 如果上一条是用药，判断用药频次是否需要日剂量相加(医嘱表达不规范的一天多次用药)，如果不需要就保留最后一条用药信息，
            if j-1 >=0:
                if (str(temp.loc[j - 1, 'DRUG_NAME']) != 'nan'):
                    ## 如果用药不需要日剂量相加，保留该条用药信息
                    if str(temp.loc[j - 1, 'FREQUENCY']) not in col_sum:
                        between_id.append(temp.iloc[[j - 1]])
                    else:
                        ## 如果上一条用药信息需要日剂量相加，继续判断前两天和前三条的平次
                        if j - 2 > 0:
                            if str(temp.loc[j - 2, 'FREQUENCY']) in col_sum:
                                temp.loc[j - 1, 'DOSAGE'] = (temp.loc[j - 1, 'DOSAGE'] + temp.loc[j - 2, 'DOSAGE']) / 2
                                temp.loc[j - 1, 'FREQUENCY'] = 'BID每日两次(自定义)'
                                if j - 3 > 0:
                                    if (str(temp.loc[j - 3, 'FREQUENCY']) in col_sum):
                                        temp.loc[j - 1, 'DOSAGE'] = (2 * temp.loc[j - 1, 'DOSAGE'] + temp.loc[
                                            j - 3, 'DOSAGE']) / 3
                                        temp.loc[j - 1, 'FREQUENCY'] = 'TID每日三次(自定义)'
                        between_id.append(temp.iloc[[j - 1]])
    # print(i)
    temp_1 = between_id[0]
    for k in range(1, len(between_id)):
        temp_1 = pd.concat([temp_1, between_id[k]], axis=0)  # 将groupby后的list类型转换成DataFrame

    temp_1.loc[:,'number'] = list(range(temp_1.shape[0]))
    temp_1 = temp_1.sort_values(['number'], ascending=0)  # 和temp排序一样，时间升序排列：一条tcms用药、一条或多条tdm检测
    temp_1 = temp_1.drop(['number'], axis=1)
    temp_1 = temp_1.reset_index()
    del temp_1['index']
    # print(temp_1[['PATIENT_ID', 'DRUG_NAME','PROJECT_NAME','time']])

    all_id.append(temp_1)

tcms_tdm_dosage = all_id[0]
for i in range(1, len(all_id)):
    tcms_tdm_dosage = pd.concat([tcms_tdm_dosage, all_id[i]], axis=0)  # 将groupby后的list类型转换成DataFrame
tcms_tdm_dosage = tcms_tdm_dosage.reset_index()
del tcms_tdm_dosage['index']

print(tcms_tdm_dosage)
print(tcms_tdm_dosage.shape)
print(np.unique(tcms_tdm_dosage['FREQUENCY'].astype('str')))


# 把频次改完数字（每天几次）
print('------------------多次用药信息dependency数字化处理----------------------------------')
one=['Q.M.每天早上服', 'QD每天', 'QD每天一次', 'QN每晚一次']
two=['BIDCXY每日两次(6,16)', 'BID一日两次', 'BID日两次', 'BID每日两次', 'BID每日两次(自定义)','Q12H间隔12小时']
three=['Q8H间隔8小时','TID3每日三次(8,16,20)', 'TID一日三次', 'TID日三次', 'TID每日三次','TID每日三次(自定义)',]
tcms_tdm_dosage['FREQUENCY']=tcms_tdm_dosage['FREQUENCY'].apply(lambda x: 1 if x in one else 2 if x in two else 3 if x in three else np.nan)

print(np.unique(tcms_tdm_dosage['FREQUENCY'].astype('str')))
tcms_tdm_dosage['日剂量']=tcms_tdm_dosage['DOSAGE']*tcms_tdm_dosage['FREQUENCY']

print(tcms_tdm_dosage.head())
print(len(np.unique(tcms_tdm_dosage['PATIENT_ID'])))


# 拆分成一条用药对应一条tdm检测的格式
print('------------------用药信息拆分成一条用药对应一条tdm检测的格式--------------------------')
all_id = []
for i in np.unique(tcms_tdm_dosage['PATIENT_ID']):
    temp = tcms_tdm_dosage[tcms_tdm_dosage['PATIENT_ID'] == i]
    temp = temp.reset_index()
    del temp['index']

    between_id = []
    for j in range(temp.shape[0]):
        ## 如果该条为用药，保留，project_name对应检测，因为drug_name不规范，用project_name更简单
        if temp.loc[j, 'PROJECT_NAME'] != '他克莫司':
            between_id.append(temp.iloc[[j]])
            ## 下一条一定为他克莫司tdm，保留，因为是他克莫司用药
            between_id.append(temp.iloc[[j + 1]])
            ## 如果有多条tdm检测，复制一下上一条用药，并保留tdm检测
            for k in range(j+2, temp.shape[0]):
                if temp.loc[k, 'PROJECT_NAME'] == '他克莫司':
                    between_id.append(temp.iloc[[j]])
                    between_id.append(temp.iloc[[k]])
                else:
                    break

    temp_1 = between_id[0]
    for m in range(1, len(between_id)):
        temp_1 = pd.concat([temp_1, between_id[m]], axis=0)  # 将groupby后的list类型转换成DataFrame
    temp_1 = temp_1.reset_index()
    del temp_1['index']

    all_id.append(temp_1)

tcms_single_tdm_single = all_id[0]
for i in range(1, len(all_id)):
    tcms_single_tdm_single = pd.concat([tcms_single_tdm_single, all_id[i]], axis=0)  # 将groupby后的list类型转换成DataFrame
tcms_single_tdm_single = tcms_single_tdm_single.reset_index()
del tcms_single_tdm_single['index']

print(tcms_single_tdm_single.head())


##   筛选初始方案中tdm符合终点事件的用药和tdm检测组合
## 筛选初始方案符合终点时间的病人id(先保留每个id的前两条信息。再判断tdm是否符合终点事件)
## 修改部分异常检测结果
print('------------------筛选初始方案中tdm符合终点事件的用药和tdm检测组合-----------------------')
tcms_single_tdm_single['TEST_RESULT']=tcms_single_tdm_single['TEST_RESULT'].astype('str').apply(lambda x: 30.00 if x=='>30.00' else float(x))
aaa= []
for i in np.unique(tcms_single_tdm_single['PATIENT_ID']):
    temp = tcms_single_tdm_single[tcms_single_tdm_single['PATIENT_ID']==i]
    temp =temp.reset_index()
    del temp['index']
    ## 第二条是tdm检测，判断一下是否符合终点事件
    if ((temp.loc[1,'time']-temp.loc[1,'SURGERY_DATE']).days<=30)&(8<=float(temp.loc[1,'TEST_RESULT'])<=15):
        aaa.append(temp.loc[:1,:])
    if (30<(temp.loc[1,'time']-temp.loc[1,'SURGERY_DATE']).days<=90)&(8<=float(temp.loc[1,'TEST_RESULT'])<=15):
        print('hello')
        aaa.append(temp.loc[:1,:])

tcms_tdm_initialCase=aaa[0]
for i in range(1,len(aaa)):
    tcms_tdm_initialCase=pd.concat([tcms_tdm_initialCase,aaa[i]],axis=0)#将groupby后的list类型转换成DataFrame
tcms_tdm_initialCase=tcms_tdm_initialCase.reset_index()
del tcms_tdm_initialCase['index']

print(tcms_tdm_initialCase)
print(len(np.unique(tcms_tdm_initialCase['PATIENT_ID']))) ## 初始方案符合终点事件的病人数为329


# 将用药和tdm检测整到同一行，计算术后移植时间和tdm检测与用药的时间间隔
print('--------------------将用药和tdm检测整到同一行-------------------------')
aaa = []
for i in np.unique(tcms_tdm_initialCase['PATIENT_ID']):
    temp = tcms_tdm_initialCase[tcms_tdm_initialCase['PATIENT_ID'] == i]
    temp = temp.reset_index()
    del temp['index']

    temp.loc[0, 'PROJECT_NAME'] = temp.loc[1, 'PROJECT_NAME']
    temp.loc[0, 'TEST_RESULT'] = temp.loc[1, 'TEST_RESULT']
    temp.loc[0, 'TDM检测时间'] = temp.loc[1, 'time']
    ## 移植术后天数=第一次服用他克莫司的时间-手术时间
    temp.loc[0, '移植术后天数'] = (temp.loc[0, 'time'] - temp.loc[0, 'SURGERY_DATE']).days
    ## tdm检测与第一次服用他克莫司的时间间隔,无法计算，因为有的服药时间间隔很长，不知道后面的多次tdm检测具体对应哪一天的服药
    # temp.loc[0,'TDM检测与服药时间间隔']=(temp.loc[1,'time']-temp.loc[0,'time']).days
    temp = temp.rename(columns={'time': '服药开始时间', 'END_DATETIME': '服药结束时间', 'SURGERY_DATE': '肾移植手术时间'})

    aaa.append(temp.iloc[[0]])

tcms_tdm_initialCase_1 = aaa[0]
for i in range(1, len(aaa)):
    tcms_tdm_initialCase_1 = pd.concat([tcms_tdm_initialCase_1, aaa[i]], axis=0)  # 将groupby后的list类型转换成DataFrame
tcms_tdm_initialCase_1 = tcms_tdm_initialCase_1.reset_index()
del tcms_tdm_initialCase_1['index']
print(tcms_tdm_initialCase_1.shape)

# 将移植术后天数小于0的修改为0,因为这种属于他克莫司长期医嘱，手术前几天开始服用，且手术当天也服用了，截取手术当天的服药时间点
tcms_tdm_initialCase_1['移植术后天数']=tcms_tdm_initialCase_1['移植术后天数'].apply(lambda x: 0 if x<0 else x)
print(np.unique(tcms_tdm_initialCase_1['移植术后天数']))

writer=pd.ExcelWriter(project_path + '/data/data_from_mysql/df_初始方案符合终点事件.xlsx')
tcms_tdm_initialCase_1.to_excel(writer)
writer.save()

print(tcms_tdm_initialCase_1)


#   筛选调整方案中tdm符合终点事件的用药和tdm检测组合（不包括初始用药）
print('-------------------筛选调整方案中tdm符合终点事件的用药和tdm检测组合-------------------------')
## 筛选调整方案符合终点时间的病人id，计算上一次tdm检测数值，上一次用药日剂量
aaa= []
for i in np.unique(tcms_single_tdm_single['PATIENT_ID']):
    temp = tcms_single_tdm_single[tcms_single_tdm_single['PATIENT_ID']==i]
    temp =temp.reset_index()
    del temp['index']
    ## 第二条是tdm检测，判断一下是否符合终点事件,排除前两条信息为初始方案的
    for j in range(2,temp.shape[0]):
        ## 如果j是奇数，即该行为tdm检测行。因为前面规范化为一条用药，一条tdm检测
        if (j % 2) != 0:
            ## 调整用药事件，移植术后天数=上一次TDM检测的时间-肾移植手术时间
            if ((temp.loc[j,'time']-temp.loc[j,'SURGERY_DATE']).days<=30)&(8<=float(temp.loc[j,'TEST_RESULT'])<=15):
                ## j-检测；j-1用药；j-2检测；j-3用药
                temp.loc[j-1,'上一次TDM值']=temp.loc[j-2,'TEST_RESULT']
                temp.loc[j-1,'上一次日剂量']=temp.loc[j-3,'日剂量']
                ## 移植术后天数=上一次TDM时间-手术时间
                temp.loc[j-1,'移植术后天数']=(temp.loc[j-2,'time']-temp.loc[0,'SURGERY_DATE']).days
                aaa.append(temp.loc[j-1:j,:])
            if (30<(temp.loc[j,'time']-temp.loc[j,'SURGERY_DATE']).days<=90)&(8<=float(temp.loc[j,'TEST_RESULT'])<=15):
                temp.loc[j-1,'上一次TDM值']=temp.loc[j-2,'TEST_RESULT']
                temp.loc[j-1,'上一次日剂量']=temp.loc[j-3,'日剂量']
                temp.loc[j-1,'移植术后天数']=(temp.loc[j-2,'time']-temp.loc[0,'SURGERY_DATE']).days
                aaa.append(temp.loc[j-1:j,:])

tcms_tdm_adjust=aaa[0]
for i in range(1,len(aaa)):
    tcms_tdm_adjust=pd.concat([tcms_tdm_adjust,aaa[i]],axis=0)#将groupby后的list类型转换成DataFrame
tcms_tdm_adjust=tcms_tdm_adjust.reset_index()
del tcms_tdm_adjust['index']
print(len(np.unique(tcms_tdm_adjust['PATIENT_ID']))) ## 调整方案符合终点事件的病人数为571)


# 将调整用药和tdm检测整到同一行，计算术后移植时间和tdm检测与用药的时间间隔
print('---------------------将调整用药和tdm检测整到同一行---------------------------')
aaa = []
for i in range(tcms_tdm_adjust.shape[0]):
    ##如果i是偶数，说明是用药行，且不允许最后一条是用药（没有相应的tdm检测）
    if (i % 2 == 0) & (i != tcms_tdm_adjust.shape[0] - 1):
        temp = tcms_tdm_adjust.loc[i:i + 1, :]
        temp = temp.reset_index()
        del temp['index']

        temp.loc[0, 'PROJECT_NAME'] = temp.loc[1, 'PROJECT_NAME']
        temp.loc[0, 'TEST_RESULT'] = temp.loc[1, 'TEST_RESULT']
        temp.loc[0, 'TDM检测时间'] = temp.loc[1, 'time']
        ## 移植术后天数=第一次服用他克莫司的时间-手术时间
        # temp.loc[0,'移植术后天数']=(tcms_tdm_adjust.loc[0,'time']-tcms_tdm_adjust.loc[0,'SURGERY_DATE']).days
        ## tdm检测与第一次服用他克莫司的时间间隔,无法计算，因为有的服药时间间隔很长，不知道后面的多次tdm检测具体对应哪一天的服药
        # temp.loc[0,'TDM检测与服药时间间隔']=(temp.loc[1,'time']-temp.loc[0,'time']).days
        temp = temp.rename(columns={'time': '服药开始时间', 'END_DATETIME': '服药结束时间', 'SURGERY_DATE': '肾移植手术时间'})

        aaa.append(temp.iloc[[0]])

tcms_tdm_adjust_1 = aaa[0]
for i in range(1, len(aaa)):
    tcms_tdm_adjust_1 = pd.concat([tcms_tdm_adjust_1, aaa[i]], axis=0)  # 将groupby后的list类型转换成DataFrame
tcms_tdm_adjust_1 = tcms_tdm_adjust_1.reset_index()
del tcms_tdm_adjust_1['index']
print(tcms_tdm_adjust_1.shape)


# 将移植术后天数小于0的修改为0,因为这种属于他克莫司长期医嘱，手术前几天开始服用，且手术当天也服用了，截取手术当天的服药时间点
tcms_tdm_adjust_1['移植术后天数']=tcms_tdm_adjust_1['移植术后天数'].apply(lambda x: 0 if x<0 else x)
print(np.unique(tcms_tdm_adjust_1['移植术后天数']))

writer=pd.ExcelWriter(project_path + '/data/data_from_mysql/df_调整方案符合终点事件.xlsx')
tcms_tdm_adjust_1.to_excel(writer)
writer.save()

#  初始方案所有变量数据清洗
print('---------------------初始方案所有变量数据清洗---------------------------')
base_gene_record['年龄']=base_gene_record['手术时间'].astype('str').apply(lambda x: int(x.split('-')[0]))-base_gene_record['年龄'].astype('str').apply(lambda x: int(x.split('-')[0]))
base_gene_record=base_gene_record.rename(columns={'性别1=12=2':'性别','病人ID':'PATIENT_ID','GYP3A5':'CYP3A5'})
base_gene_record['性别']=base_gene_record['性别'].apply(lambda x: 0 if x==1 else 1 if x==2 else x)

tcms_tdm_initialCase_1=tcms_tdm_initialCase_1.rename(columns={'TEST_RESULT':'TDM检测结果'})

print(tcms_tdm_initialCase_1.columns)
print(base_gene_record.columns)

## 计算BMI
base_gene_record['BMI']=(base_gene_record['体重']*10000)/(base_gene_record['身高']*base_gene_record['身高'])

## 合并基本信息
print('---------------------初始方案与基因型数据合并---------------------------')
tcms_tdm_initialCase_gene=pd.merge(tcms_tdm_initialCase_1[['PATIENT_ID','日剂量', 'FREQUENCY','TDM检测结果','移植术后天数','是否亲属同体']],\
                              base_gene_record[['PATIENT_ID', '性别','年龄','身高', '体重','BMI','CYP3A5','ABCB1']],\
                              on=['PATIENT_ID'],how='left')
tcms_tdm_initialCase_gene=tcms_tdm_initialCase_gene.reset_index()
del tcms_tdm_initialCase_gene['index']


print(tcms_tdm_initialCase_gene[tcms_tdm_initialCase_gene['年龄'].isnull()])
print(np.unique(tcms_tdm_initialCase_gene['ABCB1'].astype('str')))

# 基因编码0，1，2
tcms_tdm_initialCase_gene['CYP3A5']=tcms_tdm_initialCase_gene['CYP3A5'].apply(lambda x: 0 if x=='AA' else 1 if (x=='AG')|(x=='GA') else 2 if x=='GG' else x)
tcms_tdm_initialCase_gene['ABCB1']=tcms_tdm_initialCase_gene['ABCB1'].apply(lambda x: 0 if x=='CC' else 1 if (x=='CT')|(x=='TC') else 2 if x=='TT' else x)

tcms_tdm_initialCase_gene.head()


## 其他用药信息提取（酶抑制剂、康唑类、钙通道阻滞剂、拉唑类、酶诱导剂、糖皮质激素、MPA类药物）
print('---------------------提取其他用药数据---------------------------')
print(drug_record_other.columns)
# print(tcms_tdm_initialCase_1.columns)

# 只提取初始方案中的病人信息与其他用药数据进行合并
drug_record_other_1=pd.merge(tcms_tdm_initialCase_1[['PATIENT_ID','服药开始时间', '服药结束时间', '肾移植手术时间','TDM检测时间']],\
                             drug_record_other,on=['PATIENT_ID'],how='left')
print(drug_record_other_1.shape)
print(len(np.unique(drug_record_other_1['PATIENT_ID'])))

# 将其他用药结束时间缺失的修改为用药开始时间
aaa = drug_record_other_1[drug_record_other_1['END_DATETIME'].astype('str')=='0001-01-01 00:00:00']
bbb = drug_record_other_1[drug_record_other_1['END_DATETIME'].astype('str')!='0001-01-01 00:00:00']
aaa.loc[:,'END_DATETIME']=aaa.loc[:,'START_DATETIME']
drug_record_other_2=pd.concat([aaa,bbb],axis=0)
#drug_record_other_2['START_DATETIME']=drug_record_other_2['START_DATETIME'].apply(lambda x: x.to_pydatetime())
#drug_record_other_2['END_DATETIME']=drug_record_other_2['END_DATETIME'].apply(lambda x: x.to_pydatetime())
drug_record_other_2=drug_record_other_2.reset_index()
del drug_record_other_2['index']
print(drug_record_other_2.shape)

import datetime
def str_to_datatime(x):
    try:
        a=datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
        return a
    except :
        return np.NaN

# END_DATETIME有问题，要先转换成datetime格式，后面才不会报错
drug_record_other_2['END_DATETIME']=drug_record_other_2['END_DATETIME'].astype('str').apply(str_to_datatime)

# 提取第一次他克莫司服药前3天之后，第一次TDM检测之前的其他用药信息
drug_record_other_2=drug_record_other_2[(drug_record_other_2['END_DATETIME']>=drug_record_other_2['服药开始时间']-datetime.timedelta(days=3))&(drug_record_other_2['START_DATETIME']<=drug_record_other_2['TDM检测时间'])]
drug_record_other_2=drug_record_other_2.reset_index()
del drug_record_other_2['index']
print(drug_record_other_2.shape)

print(len(np.unique(drug_record_other_2['PATIENT_ID'])))
print(drug_record_other_2)

# 酶抑制剂（五酯软胶囊,日剂量）
print('---------------------五酯软胶囊--------------------------')
print(len(np.unique(drug_record_other_2[(drug_record_other_2['DRUG_NAME'].str.contains('五酯软胶囊'))]['PATIENT_ID'])))
wuzhi=drug_record_other_2[drug_record_other_2['DRUG_NAME'].str.contains('五酯软胶囊')]
print(wuzhi.shape)
print(np.unique(wuzhi['DOSAGE']))
# 统一剂量单位为mg
wuzhi.loc[:,'五酯软胶囊次剂量']=wuzhi['DOSAGE'].astype('str').apply(lambda x: '0'+x if x[0]=='.' else x)
wuzhi['五酯软胶囊次剂量']=wuzhi['五酯软胶囊次剂量'].astype('str').apply(lambda x: float(x.replace('mg',''))/1000 if 'm' in x else float(x.replace('g','')))

## 将频次修改为数字
print(np.unique(wuzhi['五酯软胶囊次剂量'].astype('str')))
one=['PRN必要时用']
two=['BIDCXY每日两次(6,16)', 'BID一日两次', 'BID日两次', 'BID每日两次','Q12H间隔12小时']
three=['Q8H间隔8小时', 'TID一日三次', 'TID日三次', 'TID每日三次']
wuzhi['FREQUENCY']=wuzhi['FREQUENCY'].apply(lambda x: 1 if x in one else 2 if x in two else 3 if x in three else np.nan)
wuzhi_filter = wuzhi[wuzhi['FREQUENCY'].astype('str') != 'nan']

# print(wuzhi.head())
print(np.unique(str(wuzhi_filter['FREQUENCY'])))
# 计算激素日剂量
wuzhi_filter['五酯软胶囊日剂量']=wuzhi_filter['五酯软胶囊次剂量']*wuzhi_filter['FREQUENCY']
## 每个病人id只保留第一次使用激素的日剂量
wuzhi_filter=wuzhi_filter.drop_duplicates(subset=['PATIENT_ID'],keep='first')
print(wuzhi_filter.shape)

# 康唑类药物（没有）
print(drug_record_other_2[(drug_record_other_2['DRUG_NAME'].str.contains('康唑'))])

# 钙通道阻滞剂
drug_record_other_2['钙通道阻滞剂']=drug_record_other_2['DRUG_NAME'].astype('str').apply(lambda x: 1 if ('地尔硫卓' in x)|('维拉帕米' in x)|('地平' in x) else 0)
print(len(np.unique(drug_record_other_2[drug_record_other_2['钙通道阻滞剂']==1]['PATIENT_ID'])))

# 拉唑类（奥美拉唑、泮托拉唑）
drug_record_other_2['拉唑类']=drug_record_other_2['DRUG_NAME'].astype('str').apply(lambda x: 1 if ('奥美拉唑' in x)|('泮托拉唑' in x) else 0)
print(len(np.unique(drug_record_other_2[drug_record_other_2['拉唑类']==1]['PATIENT_ID'])))

# 酶诱导剂（苯妥英钠、巴比妥、利福平、卡马西平、圣约翰草、贯叶连翘）
drug_record_other_2['酶诱导剂']=drug_record_other_2['DRUG_NAME'].astype('str').apply(lambda x: 1 if ('苯妥英钠' in x)|('巴比妥' in x)|('利福平' in x)|('卡马西平' in x)|('圣约翰草' in x)|('贯叶连翘' in x) else 0)
print(len(np.unique(drug_record_other_2[drug_record_other_2['酶诱导剂']==1]['PATIENT_ID'])))

# 糖皮质激素（用剂量表示）
print('---------------------糖皮质激素--------------------------')
jisu=drug_record_other_2[(drug_record_other_2['DRUG_NAME'].str.contains('甲泼尼龙'))|(drug_record_other_2['DRUG_NAME'].str.contains('泼尼松'))]
print(np.unique(drug_record_other_2[(drug_record_other_2['DRUG_NAME'].str.contains('甲泼尼龙'))]['DOSAGE']))
print(np.unique(drug_record_other_2[(drug_record_other_2['DRUG_NAME'].str.contains('泼尼松'))]['DOSAGE']))

# 统一剂量单位为mg
jisu['糖皮质激素次剂量']=jisu['DOSAGE'].astype('str').apply(lambda x: '0'+x if x[0]=='.' else x)
jisu['糖皮质激素次剂量']=jisu['糖皮质激素次剂量'].astype('str').apply(lambda x: float(x.replace('g',''))*1000 if 'm' not in x else float(x.replace('mg','')))

# 将泼尼松的剂量转换成甲泼尼龙的等价剂量，泼尼松剂量*1.25
jisu=jisu.reset_index()
del jisu['index']
for i in range(jisu.shape[0]):
    if '泼尼松' in str(jisu.loc[i,'DRUG_NAME']):
        jisu.loc[i,'糖皮质激素次剂量']=float(jisu.loc[i,'糖皮质激素次剂量'])*1.25

# 将频次修改为数字
one=['QD每天', 'QD每天一次','PRN必要时用','ST立即用']
two=['BID每日两次', 'Q.M.每天早上服','QN每晚一次']
three=['TH.顿服']
jisu['FREQUENCY']=jisu['FREQUENCY'].apply(lambda x: 1 if x in one else 2 if x in two else 3 if x in three else np.nan)
print(np.unique(str(jisu['FREQUENCY'])))
print(jisu['FREQUENCY'].isnull().sum())
jisu_filter = jisu[jisu['FREQUENCY'].astype('str') != 'nan']

# 计算激素日剂量
jisu_filter['糖皮质激素日剂量']=jisu_filter['糖皮质激素次剂量']*jisu_filter['FREQUENCY']
## 每个病人id只保留第一次使用激素的日剂量
jisu_filter=jisu_filter.drop_duplicates(subset=['PATIENT_ID'],keep='first')
print(jisu_filter.shape)
print(len(np.unique(jisu_filter['PATIENT_ID'])))

# MPA类药物（霉酚酸、吗替麦考酚酯,剂型，四分类（包括没有使用）
print('---------------------MPA类药物--------------------------')
drug_record_other_2['MPA类药物']=drug_record_other_2['DRUG_NAME'].astype('str').apply(lambda x: 1 if ('吗替麦考酚酯' in x)|('霉酚酸' in x) else 0)
print(len(np.unique(drug_record_other_2[drug_record_other_2['MPA类药物']==1]['PATIENT_ID'])))

print(drug_record_other_2.columns)

# 将其他用药信息合并到初始建模数据集中,五酯软胶囊、糖皮质激素日剂量、mpa类药物的缺失值为0
print('---------------------将其他用药信息合并到初始建模数据集--------------------------')
tcms_tdm_initialCase_gene_1=pd.merge(tcms_tdm_initialCase_gene,wuzhi_filter[['PATIENT_ID','五酯软胶囊日剂量']], on=['PATIENT_ID'],how='left')
tcms_tdm_initialCase_gene_1=pd.merge(tcms_tdm_initialCase_gene_1,drug_record_other_2[drug_record_other_2['钙通道阻滞剂']==1].drop_duplicates(subset=['PATIENT_ID'],keep='first')[['PATIENT_ID','钙通道阻滞剂']],\
                                on=['PATIENT_ID'],how='left')
tcms_tdm_initialCase_gene_1=pd.merge(tcms_tdm_initialCase_gene_1,drug_record_other_2[drug_record_other_2['拉唑类']==1].drop_duplicates(subset=['PATIENT_ID'],keep='first')[['PATIENT_ID','拉唑类']],\
                                on=['PATIENT_ID'],how='left')
tcms_tdm_initialCase_gene_1=pd.merge(tcms_tdm_initialCase_gene_1,drug_record_other_2[drug_record_other_2['酶诱导剂']==1].drop_duplicates(subset=['PATIENT_ID'],keep='first')[['PATIENT_ID','酶诱导剂']],\
                                on=['PATIENT_ID'],how='left')
tcms_tdm_initialCase_gene_1=pd.merge(tcms_tdm_initialCase_gene_1,drug_record_other_2[drug_record_other_2['MPA类药物']==1].drop_duplicates(subset=['PATIENT_ID'],keep='first')[['PATIENT_ID','MPA类药物']],\
                                on=['PATIENT_ID'],how='left')
tcms_tdm_initialCase_gene_1=pd.merge(tcms_tdm_initialCase_gene_1,jisu_filter[['PATIENT_ID','糖皮质激素日剂量']], on=['PATIENT_ID'],how='left')

# 药物是否使用，缺失值插补0，即没有使用
col_name=['五酯软胶囊日剂量', '钙通道阻滞剂', '拉唑类', '酶诱导剂','糖皮质激素日剂量','MPA类药物']
for i in col_name:
    tcms_tdm_initialCase_gene_1[i]=tcms_tdm_initialCase_gene_1[i].replace(np.nan,0)

print(tcms_tdm_initialCase_gene_1.shape)

# 检验信息提取合并入初始方案中
print('-----------------------检验信息提取合并入初始方案建模数据中--------------------------')
print(test_record_result.columns)
## 只提取初始方案中的病人检测结果信息
initialCase_test_result=pd.merge(tcms_tdm_initialCase_1[['PATIENT_ID','服药开始时间', '服药结束时间', '肾移植手术时间','TDM检测时间']],\
                       test_record_result,on=['PATIENT_ID'],how='left')
print(initialCase_test_result.shape)

# 截取手术之后，第一次用药之前的检验信息
initialCase_test_result_1=initialCase_test_result[(initialCase_test_result['COLLECT_TIME']>=initialCase_test_result['肾移植手术时间'])&(initialCase_test_result['COLLECT_TIME']<=initialCase_test_result['服药开始时间'])]
initialCase_test_result_1=initialCase_test_result_1.reset_index()
del initialCase_test_result_1['index']
print(initialCase_test_result_1.shape)

# 同一个病人ID的相同检验项，取最接近服药的那一次检验
initialCase_test_result_1=initialCase_test_result_1.drop_duplicates(subset=['PATIENT_ID','PROJECT_NAME'],keep='last')
initialCase_test_result_1=initialCase_test_result_1.reset_index()
del initialCase_test_result_1['index']
print(initialCase_test_result_1.shape)

# 有5个病人没有检验信息
print(len(np.unique(initialCase_test_result_1['PATIENT_ID'])))


print(initialCase_test_result_1[initialCase_test_result_1['PROJECT_NAME'].str.contains('红细胞')]['PROJECT_NAME'].unique())
print(initialCase_test_result_1[initialCase_test_result_1['PROJECT_NAME'].str.contains('血红蛋白')]['PROJECT_NAME'].unique())

# 肾功能*（尿酸、肌酐）、肝功能（谷草、总胆红素）、血常规（红细胞压积）、中性粒细胞%（或中性粒细胞百分数）、淋巴细胞
initialCase_test_result_1['PROJECT_NAME']=initialCase_test_result_1['PROJECT_NAME'].replace('中性粒细胞%','中性粒细胞百分数')
initialCase_test_result_1['PROJECT_NAME']=initialCase_test_result_1['PROJECT_NAME'].replace('中性粒细胞百分数 ','中性粒细胞百分数')
initialCase_test_result_1['PROJECT_NAME']=initialCase_test_result_1['PROJECT_NAME'].replace('淋巴细胞%','淋巴细胞百分数')
initialCase_test_result_1['PROJECT_NAME']=initialCase_test_result_1['PROJECT_NAME'].replace('红细胞','红细胞计数')
initialCase_test_result_1['PROJECT_NAME']=initialCase_test_result_1['PROJECT_NAME'].replace('红细胞数','红细胞计数')
initialCase_test_result_1['PROJECT_NAME']=initialCase_test_result_1['PROJECT_NAME'].replace('红细胞计数 ','红细胞计数')
initialCase_test_result_2=initialCase_test_result_1[(initialCase_test_result_1['PROJECT_NAME'].str.contains('尿酸'))|(initialCase_test_result_1['PROJECT_NAME'].str.contains('肌酐'))|\
                            (initialCase_test_result_1['PROJECT_NAME'].str.contains('谷草转氨酶'))|(initialCase_test_result_1['PROJECT_NAME'].str.contains('总胆红素'))|\
                            (initialCase_test_result_1['PROJECT_NAME'].str.contains('红细胞压积'))|(initialCase_test_result_1['PROJECT_NAME']=='红细胞计数')|\
                            (initialCase_test_result_1['PROJECT_NAME']=='血红蛋白')|\
                            (initialCase_test_result_1['PROJECT_NAME'].str.contains('中性粒细胞百分数'))|(initialCase_test_result_1['PROJECT_NAME'].str.contains('淋巴细胞百分数'))]
initialCase_test_result_2=initialCase_test_result_2.reset_index()
del initialCase_test_result_2['index']
print(initialCase_test_result_2.shape)

# 检验项转置为表头
initialCase_test_result_2['TEST_RESULT']=initialCase_test_result_2['TEST_RESULT'].apply(judge_float)
initialCase_test_result_3= initialCase_test_result_2.pivot_table('TEST_RESULT', ['PATIENT_ID'], 'PROJECT_NAME')
initialCase_test_result_3=initialCase_test_result_3.reset_index()

# 把检验项合并到建模数据集中
initialCase_test_result_merge=pd.merge(tcms_tdm_initialCase_1,initialCase_test_result_3,on=['PATIENT_ID'],how='left')
initialCase_test_result_merge=initialCase_test_result_merge.reset_index()
del initialCase_test_result_merge['index']
print(initialCase_test_result_merge.shape)

print(initialCase_test_result_merge.columns)
print(initialCase_test_result_merge.head())

# 诊断信息提取（生理病理状况（腹泻、腹痛。。。）、高血压、糖尿病、幽门螺旋杆菌感染）
print(print('---------------------诊断信息提取并入初始方案建模数据--------------------------'))
print(diagnose_record.columns)

## 只提取初始方案中的病人信息
diagnose_record_1=pd.merge(tcms_tdm_initialCase_1[['PATIENT_ID']],diagnose_record,on=['PATIENT_ID'],how='left')
print(diagnose_record_1.shape)

diagnose_record_1['高血压']=diagnose_record_1['DIAGNOSTIC_CONTENT'].apply(lambda x: 1 if '高血压' in x else 0)
diagnose_record_1['糖尿病']=diagnose_record_1['DIAGNOSTIC_CONTENT'].apply(lambda x: 1 if '糖尿病' in x else 0)
diagnose_record_1['生理病理状况']=diagnose_record_1['DIAGNOSTIC_CONTENT'].apply(lambda x: 1 if ('腹泻' in x)|('腹痛' in x)|('腹胀' in x)|('嗳气' in x)|('纳差' in x)|('胃肠动力恢复' in x) else 0)
#diagnose_record_1['幽门螺旋杆菌感染']=diagnose_record_1['DIAGNOSTIC_CONTENT'].apply(lambda x: 1 if '幽门螺旋杆菌' in x else 0)

print(diagnose_record_1[diagnose_record_1['糖尿病']==1].shape)
print(diagnose_record_1.columns)

## 把检验项合并到建模数据集中
initialCase_test_result_diagnose=pd.merge(initialCase_test_result_merge,diagnose_record_1[['PATIENT_ID','高血压','糖尿病','生理病理状况']],on=['PATIENT_ID'],how='left')
initialCase_test_result_diagnose=initialCase_test_result_diagnose.reset_index()
del initialCase_test_result_diagnose['index']
print(initialCase_test_result_diagnose.shape)
## 
initialCase_test_result_diagnose=initialCase_test_result_diagnose.rename(columns={'是否亲属同体':'是否亲属活体'})
print(initialCase_test_result_diagnose.columns)
print(initialCase_test_result_diagnose.head())


writer=pd.ExcelWriter(project_path + '/data/data_from_mysql/df_初始方案建模数据集.xlsx')
initialCase_test_result_diagnose.to_excel(writer)
writer.save()