# _*_ coding: utf-8 _*_
# @Time: 2021/9/309:36 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description: 他克莫司DM项目学习：1. 从数据库中提取数据

import pymysql as MySQLDB
import pandas as pd
import numpy as np

import sys
import os

project_path = os.getcwd()

conn = MySQLDB.connect(host='localhost', port=3306, user='root', password='123456', db='mdnov_ciip_ipharma_zdy', charset='UTF8')
cursor = conn.cursor()
cursor.execute("Select version()")
for i in cursor:
    print(i)


# 手术表单
#读取数据
try:
    sql = 'select PATIENT_ID, SURGERY_CODE,SURGERY_NAME,SURGERY_DATE from surgical_record;'
    surgical_record = pd.read_sql(sql, conn)
except MySQLDB.err.ProgrammingError as e:
    print('surgical_record Error is ' + str(e))
    sys.exit()

print(surgical_record.shape)
print(np.unique(surgical_record[(surgical_record['SURGERY_NAME'].str.contains('肾'))&((surgical_record['SURGERY_NAME'].str.contains('活体'))|(surgical_record['SURGERY_NAME'].str.contains('异体')))]['SURGERY_NAME']))

# 抽取肾移植患者：肾
surgical_record_kidney = surgical_record[(surgical_record['SURGERY_NAME'].str.contains('肾'))&((surgical_record['SURGERY_NAME'].str.contains('活体'))|(surgical_record['SURGERY_NAME'].str.contains('异体')))]
# print(surgical_record_kidney)

# 排除多器官联合移植的患者（肝移植、肺移植、心脏移植、骨髓移植）这些都没有
# print(np.unique(surgical_record_kidney[(surgical_record_kidney['SURGERY_NAME'].str.contains('心脏'))]['SURGERY_NAME']))
print(np.unique(surgical_record_kidney[(surgical_record_kidney['SURGERY_NAME'].str.contains('肝'))|(surgical_record_kidney['SURGERY_NAME'].str.contains('肺'))|(surgical_record_kidney['SURGERY_NAME'].str.contains('心脏'))|(surgical_record_kidney['SURGERY_NAME'].str.contains('骨髓'))]['SURGERY_NAME']))

print(surgical_record_kidney.shape)
print(len(np.unique(surgical_record_kidney['PATIENT_ID'])))


# 排除二次移植患者
print('------------------------排除二次移植患者-------------------------')
aaa=[]
for i in np.unique(surgical_record_kidney['PATIENT_ID']):
    temp=surgical_record_kidney[surgical_record_kidney['PATIENT_ID']==i]
    if temp.shape[0]>1:
        print(i)
        aaa.append(i)
# print(aaa)
print(len(aaa))
surgical_record_kidney['PATIENT_ID'] = surgical_record_kidney['PATIENT_ID'].apply(lambda x: np.nan if x in aaa else x)
surgical_record_kidney_first = surgical_record_kidney[surgical_record_kidney['PATIENT_ID'].notnull()]
surgical_record_kidney_first = surgical_record_kidney_first.reset_index()
del surgical_record_kidney_first['index']
print(surgical_record_kidney_first.shape)
print(len(np.unique(surgical_record_kidney_first['PATIENT_ID'])))
# print(surgical_record_kidney_first)


# 判断手术肾源是否亲属同体，同体活体为1，异体为0
print('------------------------判断手术肾源是否同体-------------------------')
surgical_record_kidney_first['是否亲属同体']=surgical_record_kidney_first['SURGERY_NAME'].apply(lambda x: 1 if '活体' in x else 0 if '异体' in x else x)
print(surgical_record_kidney_first)
print(np.unique(surgical_record_kidney_first['是否亲属同体']))
surgical_record_kidney_first['PATIENT_ID'] = surgical_record_kidney_first['PATIENT_ID'].astype('str')
print(surgical_record_kidney_first.shape)

writer=pd.ExcelWriter(project_path + '/data/data_from_mysql/df_肾移植手术时间.xlsx')
surgical_record_kidney_first.to_excel(writer)
writer.save()


# 用药医嘱表单
print('------------------------用药医嘱表单-------------------------')
#读取数据
try:
    sql = 'select PATIENT_ID,LONG_D_ORDER,DRUG_CODE,DRUG_NAME,AMOUNT,DRUG_SPEC,DOSAGE,FREQUENCY,MEDICATION_WAY,DRUG_INTERVAL,DURATION,START_DATETIME,END_DATETIME,CONFIRMTIME,DRUG_FLAG from doctor_order where DRUG_FLAG="1";'
    drug_record = pd.read_sql(sql, conn)
except MySQLDB.err.ProgrammingError as e:
    print('drug_record Error is ' + str(e))
    sys.exit()
# print('数据库连接成功')
print(drug_record)
print(np.unique(drug_record['CONFIRMTIME'].astype('str')))
print(np.unique((drug_record[(drug_record['DRUG_NAME'].str.contains("他克莫司"))&(drug_record['LONG_D_ORDER']=='1')]['FREQUENCY'])))
# 筛选他克莫司的长期医嘱
drug_record_tcms_1=drug_record[(drug_record['DRUG_NAME'].str.contains("他克莫司"))&(drug_record['LONG_D_ORDER']=='1')]
# 删除用药频次不明确的医嘱信息（比如：遵 医嘱遵医嘱、BY 备用、ST 立即使用、PRN 必要时用)
drug_record_tcms_2=drug_record_tcms_1[(drug_record_tcms_1['FREQUENCY']!='遵医嘱遵医嘱')&(drug_record_tcms_1['FREQUENCY']!='BY备用')&(drug_record_tcms_1['FREQUENCY']!='PRN必要时用')&(drug_record_tcms_1['FREQUENCY']!='ST立即用')]
# 删除用药剂量为 0 的医嘱信息
drug_record_tcms_3=drug_record_tcms_2[drug_record_tcms_2['DOSAGE']!='0']
print(drug_record_tcms_3.shape)

drug_record_tcms_3=drug_record_tcms_3.reset_index()
del drug_record_tcms_3['index']
print(drug_record_tcms_3.head())
print(len(np.unique(drug_record_tcms_3['PATIENT_ID'])))


# 提取做过肾移植且使用他克莫司的病人信息
print('-----------------------提取做过肾移植并使用他克莫司的病人-------------------------')
drug_record_tcms_3['PATIENT_ID']=drug_record_tcms_3['PATIENT_ID'].astype('str')
surgical_record_kidney_first['PATIENT_ID']=surgical_record_kidney_first['PATIENT_ID'].astype('str')
drug_tcms_surgery_merge=pd.merge(drug_record_tcms_3,surgical_record_kidney_first[['PATIENT_ID']],on=['PATIENT_ID'],how='inner')
drug_tcms_surgery_merge=drug_tcms_surgery_merge.reset_index()
del drug_tcms_surgery_merge['index']
print(drug_tcms_surgery_merge)
print(drug_tcms_surgery_merge.shape)
print(len(np.unique(drug_tcms_surgery_merge['PATIENT_ID'])))

writer=pd.ExcelWriter(project_path + '/data/data_from_mysql/df_他克莫司用药数据.xlsx')
drug_tcms_surgery_merge.to_excel(writer)
writer.save()


# 提取使用他克莫司手术病人的其他用药信息
print('-----------------------提取使用他克莫司手术病人的其他用药信息-------------------------')
# 提取其他用药的数据
drug_record_other = drug_record[(drug_record['DRUG_FLAG']=='1')&(drug_record['DOSAGE']!='0')]
drug_record_other = drug_record_other.reset_index()
del drug_record_other['index']
print(drug_record_other.shape)

# 只保留用了他克莫司的肾移植病人的其他用药信息
drug_record_other['PATIENT_ID'] = drug_record_other['PATIENT_ID'].astype('str')
ttt = pd.DataFrame(np.unique(drug_tcms_surgery_merge['PATIENT_ID'])).rename(columns={0:'PATIENT_ID'})
drug_record_tcms_other = pd.merge(drug_record_other,ttt,on=['PATIENT_ID'],how='inner')

drug_record_tcms_other = drug_record_tcms_other[drug_record_tcms_other['DRUG_NAME']!='煎药费(人工)']
print(drug_record_tcms_other)
print(drug_record_tcms_other.shape)
print(len(np.unique(drug_record_tcms_other['PATIENT_ID'])))

drug_record_tcms_other=drug_record_tcms_other.reset_index()
del drug_record_tcms_other['index']

writer=pd.ExcelWriter(project_path + '/data/data_from_mysql/df_其他用药数据.xlsx')
drug_record_tcms_other.to_excel(writer)
writer.save()


# 检验表单test_record
print('-----------------------检验记录表单-------------------------')
try:
    sql = 'select PATIENT_ID,TEST_RECORD_ID,COLLECT_TIME from test_record;'
    test_record = pd.read_sql(sql, conn)
except MySQLDB.err.ProgrammingError as e:
    print('test_record Error is ' + str(e))
    sys.exit()

print(test_record.shape)
test_record['PATIENT_ID']=test_record['PATIENT_ID'].astype('str')

ttt=pd.DataFrame(np.unique(drug_tcms_surgery_merge['PATIENT_ID'])).rename(columns={0:'PATIENT_ID'})
test_record_tcms=pd.merge(test_record,ttt,on=['PATIENT_ID'],how='inner')
test_record_tcms=test_record_tcms.reset_index()
del test_record_tcms['index']
print(test_record_tcms.shape)
print(len(np.unique(test_record_tcms['TEST_RECORD_ID'])))

# 相同检验id保留第一个
test_record_tcms=test_record_tcms.drop_duplicates(subset=['TEST_RECORD_ID'],keep='first')
test_record_tcms=test_record_tcms.reset_index()
del test_record_tcms['index']
print(test_record_tcms.shape)

writer=pd.ExcelWriter(project_path + '/data/data_from_mysql/df_检验序号索引.xlsx')
test_record_tcms.to_excel(writer)
writer.save()


# 检验结果表单test_result
print('-----------------------检验结果表单-------------------------')
#读取数据
try:
    sql = 'select TEST_RECORD_ID,PROJECT_CODE,PROJECT_NAME,TEST_RESULT,RESULT_UNIT,IS_NORMAL,REFER_SCOPE from test_result;'
    test_result = pd.read_sql(sql, conn)
except MySQLDB.err.ProgrammingError as e:
    print('test_result Error is ' + str(e))
    sys.exit()

test_result_tcms=pd.merge(test_record_tcms,test_result,on=['TEST_RECORD_ID'],how='inner')
test_result_tcms=test_result_tcms.reset_index()
del test_result_tcms['index']
print(test_result_tcms.shape)
print(len(np.unique(test_result_tcms['TEST_RECORD_ID'])))
# print(len(np.unique(test_result_tcms['PATIENT_ID'])))

# 检验结果按照病人id排序
test_result_tcms_2=test_result_tcms.sort_values(['PATIENT_ID'],ascending=1)
print(len(np.unique(test_result_tcms_2['PATIENT_ID'])))
print(test_result_tcms_2.shape)

'''
test_result数据量太大，无法保存出来
writer=pd.ExcelWriter(project_path + '/data/data_from_mysql/df_检验数据.xlsx')
test_result_tcms_2.to_excel(writer)
writer.save()
'''


# 诊断表单
print('-----------------------诊断记录表单-------------------------')
#读取数据
try:
    sql = 'select PATIENT_ID,DIAGNOSTIC_CONTENT from diagnostic_record;'
    diagnostic_record = pd.read_sql(sql, conn)
except MySQLDB.err.ProgrammingError as e:
    print('diagnostic_record Error is ' + str(e))
    sys.exit()

print(diagnostic_record.shape)

# 只要肾移植后服用他克莫司的病人
#drug_record_2_merge = pd.read_excel(u'D:/aaa诺道医学/他克莫司/数据库提取的数据/df_他克莫司用药数据.xlsx')
diagnostic_record['PATIENT_ID']=diagnostic_record['PATIENT_ID'].astype('str')
ttt=pd.DataFrame(np.unique(drug_tcms_surgery_merge['PATIENT_ID'])).rename(columns={0:'PATIENT_ID'})
diagnostic_record_tcms=pd.merge(diagnostic_record,ttt,on=['PATIENT_ID'],how='inner')

print(diagnostic_record_tcms.shape)
print(len(np.unique(diagnostic_record_tcms['PATIENT_ID'])))

# 同一个病人的相同诊断，去重
diagnostic_record_tcms_dd = diagnostic_record_tcms.drop_duplicates(subset=['PATIENT_ID','DIAGNOSTIC_CONTENT'],keep='first')
diagnostic_record_tcms_dd=diagnostic_record_tcms_dd.reset_index()
del diagnostic_record_tcms_dd['index']
print(diagnostic_record_tcms_dd.shape)
print(len(np.unique(diagnostic_record_tcms_dd['PATIENT_ID'])))

diagnose = diagnostic_record_tcms_dd[['PATIENT_ID','DIAGNOSTIC_CONTENT']].groupby(['PATIENT_ID'])['DIAGNOSTIC_CONTENT'].apply(lambda x: '；'.join(x)).reset_index()
print(diagnose.shape)
print(diagnose.head())

writer=pd.ExcelWriter(project_path + '/data/data_from_mysql/df_诊断数据.xlsx')
diagnose.to_excel(writer)
writer.save()
