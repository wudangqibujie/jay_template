import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
import seaborn as sns
p = sns.color_palette()
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Arial']})


df_loan_time_train = pd.read_csv("../data/rong360/train/loan_time_train.txt", header=None, names=['用户标识', '放款时间'])
df_loan_time_train['放款时间'] = df_loan_time_train['放款时间'] // 86400
df_user_info_train = pd.read_csv('../data/rong360/train/user_info_train.txt', header=None, names=['用户标识',
                                                                                                  '用户性别',
                                                                                                  '用户职业',
                                                                                                  '用户教育程度',
                                                                                                  '用户婚姻状态',
                                                                                                  '用户户口类型'])
bill_columns = ['用户标识',
                '时间',
                '银行标识',
                '上期账单金额',
                '上期还款金额',
                '信用卡额度',
                '本期账单余额',
                '本期账单最低还款额',
                '消费笔数',
                '本期账单金额',
                '调整金额',
                '循环利息',
                '可用余额',
                '预借现金额度',
                '还款状态']
df_bill_train = pd.read_csv('../data/rong360/train/bill_detail_train.txt', header=None, names=bill_columns)
df_bill_train['时间'] = df_bill_train['时间'] // 86400
df_bill_train = pd.merge(df_bill_train, df_loan_time_train, on='用户标识', how='inner')
df_browse_train = pd.read_csv("../data/rong360/train/browse_history_train.txt", header=None, names=['用户标识',
                                                                                                    '浏览时间',
                                                                                                    '浏览行为数据',
                                                                                                    '浏览子行为编号'])
df_browse_train['浏览时间'] = df_browse_train['浏览时间'] // 86400
df_browse_train = pd.merge(df_browse_train, df_loan_time_train, on='用户标识', how='inner')
df_bank_detail_train = pd.read_csv("../data/rong360/train/bank_detail_train.txt").rename(index=str,
                                                                                         columns={"uid": "用户标识",
                                                                                                  "timespan": "流水时间",
                                                                                                  "type": "交易类型",
                                                                                                  "amount": "交易金额",
                                                                                                  "markup": "工资收入标记"})
df_bank_detail_train['流水时间'] = df_bank_detail_train['流水时间'] // 86400
df_train = pd.read_csv("../data/rong360/train/overdue_train.txt", header=None, names=['用户标识', '标签'])

df_train = pd.merge(df_train, df_user_info_train, how='inner',on = "用户标识")
df_train.to_csv('../data/rong360/train/train.csv', index=False, encoding='utf-8')

df_train = pd.read_csv('../data/rong360/train/train.csv')
print(df_bill_train.shape)

d = df_bill_train[df_bill_train["时间"] > 0]
print(d.shape)

