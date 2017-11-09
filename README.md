# dynamic-SVM



# -*- coding: utf-8 -*-

"""

Created on Thu Nov  9
15:52:08 2017

 

@author: 44091781

"""

 

#%%(1)技术准备  

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn import svm

 

#%%(2)清洗数据    

path_col_data = '/Users/44091781/Desktop/Python Test/Machine
Learning/50-index/50index_col-2006.xlsx'

path_row_data = '/Users/44091781/Desktop/Python Test/Machine
Learning/50-index/50index_row-2006.xlsx'

df_col_data = pd.read_excel(path_col_data,sheetname =
"Sheet1")

df_row_data = pd.read_excel(path_row_data,sheetname =
"Sheet1")

num = 50 

 

col_data = df_col_data.iloc[:,:num]

list = col_data.columns

length = []

for i in range(num):

    check_data =
col_data.iloc[:,i:i+1].dropna()

   
length.append(len(check_data))

choose = pd.DataFrame([np.nan] * np.ones((num,2)))

choose.iloc[:,0] = list.T

choose.iloc[:,1] = length

choose.columns = ['stock','length']

choose.index = choose.iloc[:,1]

choose = choose.sort_index(axis=0,ascending=False) #by?

long_choose = int(num*0.8)

choose_list = choose.iloc[:long_choose,0]

                

#%% （3）  traindata 整理  

row_data  =
df_row_data[(df_row_data['简称']==choose_list.iloc[0])]

for i in range(long_choose-1):

    row_data2 =
df_row_data[(df_row_data['简称']==choose_list.iloc[i+1])]

    row_data =
pd.concat([row_data, row_data2],axis=0)

''' 上一个文档 删除5，7，8 保留4流通股本，6市盈率 '''

train_data =
pd.concat([row_data.iloc[:,0:8],row_data.iloc[:,9:10]],axis=1)

train_data.index = train_data.iloc[:,0]

train_data = train_data.iloc[:,1:]

 

#将  流通股票 和 成交量 换成 环比，观测是否可以减少运算量 。

def label_c_s(data_factor,i):

    train_df =
data_factor[(data_factor['简称']==i)]

    train_df['A股流通股本(股)change']
= train_df['A股流通股本(股)']/train_df['A股流通股本(股)'].shift(1)

    train_df['成交量(股)change']
= train_df['成交量(股)']/train_df['成交量(股)'].shift(1)   

    train_df['return']
= np.log(train_df.iloc[:,2]/train_df.iloc[:,2].shift(1))

   
train_df['return-next'] = train_df['return'].shift(-1)

    train_df =
train_df.iloc[:-1,:]

   
train_df['status-price'] = 1

    judge =
train_df.iloc[:,2].fillna(9999)

    for i in
range(len(judge)):

        if
judge.iloc[i] == 999:

           
train_df['status'][i] = 0

    return train_df

 

final_data = label_c_s(train_data,choose_list.iloc[0])

 

for i in choose_list:

    if i ==
choose_list.iloc[0]:

        final_data =
label_c_s(train_data,choose_list.iloc[0])

    else:

        final_data2 =
label_c_s(train_data,i)

        final_data =
pd.concat([final_data,final_data2],axis=0)

final_data = final_data.sort_index(axis=0,ascending=True)

 

#%% (4)开始训练 动态 调整

class Para:

    method = 'svm'

    percent_select =
[0.3, 0.3]

    percent_cv = 0.1

    seed = 42

    svm_kernel =
'linear'

    svm_c = 0.01 #SVM惩罚系数C

para = Para()

 

def label_data(data):

    data['return_bin']
= np.nan

    data =
data.sort_values (by= 'return-next' , ascending = False)

    n_stock_positive=
int(para.percent_select[0]*len(data))

    n_stock_negative=
-int(para.percent_select[1]*len(data))

#    n_stock_select=
np.multiply(para.percent_select,data,shape[0])

    data.iloc[0:
n_stock_positive,-1] = 1

   
data.iloc[n_stock_negative:,-1] = 0

    data =
data.dropna(axis=0)

    return data

 

long = len(row_data)

date_long  =  len(df_col_data.index)

test = 0.1

T = int((1-test*2)*date_long)

start = int(date_long*test)

start0 = start+T

date_index = df_col_data.index[-start0:-start]

data_curr_month = final_data.iloc[0:1,:]

data_curr_month['return_bin'] = np.nan

for i_month in date_index:

    data_curr_month2 =
final_data[(final_data.index == i_month)].dropna(axis = 0)

    data_curr_month2 =
label_data(data_curr_month2)

    data_curr_month =
pd.concat([data_curr_month,data_curr_month2],axis=0)

data_in_sample = 
data_curr_month.iloc[1:,:]   

 

X_in_sample = pd.concat([data_in_sample.iloc[:,4:6],data_in_sample.iloc[:,7:10]],axis=1)    

y_in_sample = data_in_sample.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train,X_cv,y_train,y_cv=train_test_split(X_in_sample,y_in_sample,test_size=para.percent_cv,random_state=para.seed)

 

from sklearn import decomposition #主成分分析

pca = decomposition.PCA(n_components = 0.95)

pca.fit(X_train)

X_train = pca.transform(X_train)

X_cv = pca.transform(X_cv)

 

'''SVM'''

if para.method =='SVM':

    from sklearn
import svm

    model =
svm.SVC(kernel = para.svm_kernel,C=para.svm_c)

if para.method == 'SVM':

   
model.fit(X_train,y_train)

    y_pred_train =
model.predict(X_train)

    y_score_train =
model.decision_function(X_train)

    y_pred_cv =
model.predict(X_cv)

    y_score_cv =
model.decision_function(X_cv)

 

'''SVM 训练'''

model = svm.SVC(kernel = para.svm_kernel,C=para.svm_c)    

model.fit(X_train,y_train)   


y_pred_train = model.predict(X_train)

y_score_train = model.decision_function(X_train)  

y_pred_cv = model.predict(X_cv)  

y_score_cv = model.decision_function(X_cv)

#%%

'''创建三个空数据集y_true_test、y_pred_test 和y_score_test'''

y_true_test = pd.DataFrame([np.nan] *
np.ones((long_choose,1)))

y_true_test.index = choose_list

y_pred_test = pd.DataFrame([np.nan] * np.ones((long_choose,1)))

y_pred_test.index = choose_list

y_score_test = pd.DataFrame([np.nan] *
np.ones((long_choose,1)))

y_score_test.index = choose_list

 

#因子NAN也算停牌，status只是价格停牌

date_index_test = df_col_data.index[-start:-1]

 

for i_month in date_index_test:

    data_curr_month =
final_data[(final_data.index == i_month)]

   
data_curr_month_dropna = data_curr_month.dropna(axis = 0)

    X_curr_month =
pd.concat([data_curr_month_dropna.iloc[:,4:6],data_curr_month_dropna.iloc[:,7:10]],axis
= 1)

    X_curr_month =
pca.transform(X_curr_month)

    

    y_pred_curr_month
= model.predict(X_curr_month)

    y_score_curr_month
= model.decision_function(X_curr_month)

    

    y_true_df =
data_curr_month['return']

    y_true_df.index =
data_curr_month['简称']

    y_pred_df = pd.DataFrame(y_pred_curr_month)

    y_pred_df.index =
data_curr_month_dropna['简称']

    y_score_df =
pd.DataFrame(y_score_curr_month)

    y_score_df.index =
data_curr_month_dropna['简称']

    

    y_true_test =
pd.concat([y_true_test,y_true_df],axis=1)

    y_pred_test =
pd.concat([y_pred_test,y_pred_df],axis=1)

    y_score_test =
pd.concat([y_score_test,y_score_df],axis=1)

 

y_true_test = y_true_test.iloc[:,1:]

y_pred_test = y_pred_test.iloc[:,1:]

y_score_test = y_score_test.iloc[:,1:]

 

#%% (6)评价

print('training set, accuracy =
%.2f'%metrics.accuracy_score(y_train, y_pred_train))

print('training set, AUC =
%.2f'%metrics.roc_auc_score(y_train, y_score_train))

print('cv set, accuracy = %.2f'%metrics.accuracy_score(y_cv,
y_pred_cv))

print('cv set, AUC = %.2f'%metrics.roc_auc_score(y_cv,
y_score_cv))

 

y_true_total = pd.DataFrame([np.nan] * np.ones((1,1)))

y_pred_total = pd.DataFrame([np.nan] * np.ones((1,1)))

y_score_total = pd.DataFrame([np.nan] * np.ones((1,1)))

 

a= 0

for i_month in date_index_test:    

    data_curr_month =
final_data[(final_data.index == i_month)]

   
data_curr_month_dropna = data_curr_month.dropna(axis = 0)

   
data_curr_month_dropna.index = data_curr_month_dropna['简称']

    y_curr_month =
label_data(data_curr_month_dropna)['return_bin']

 

    data_curr_month =
final_data.iloc[i_month*long_choose:i_month*long_choose+long_choose,:]        

   
data_curr_month_dropna = data_curr_month.dropna(axis = 0)

   
data_curr_month_dropna.index = data_curr_month_dropna['简称']

    y_curr_month =
label_data(data_curr_month_dropna)['return_bin']

    

    y_pred_curr_month
= y_pred_test.iloc[:,a]

    y_score_curr_month
= y_score_test.iloc[:,a]

 

    y_pred_curr_month
= y_pred_curr_month[y_curr_month.index]

    y_score_curr_month
= y_score_curr_month[y_curr_month.index]

    

    y_true_total =
pd.concat([y_true_total,y_curr_month],axis=0)

    y_pred_total =
pd.concat([y_pred_total,y_pred_curr_month],axis=0)

    y_score_total =
pd.concat([y_score_total,y_score_curr_month],axis=0)

    a = a+1

    

    print('test set,
month %d, accuracy = %.2f'%(i_month,metrics.accuracy_score(y_curr_month,
y_pred_curr_month)))    

    print('test set,
month %d, AUC = %.2f'%(i_month,
metrics.roc_auc_score(y_curr_month,y_score_curr_month)))

 

y_true_total = y_true_total.iloc[1:,:]

y_pred_total = y_pred_total.iloc[1:,:]

y_score_total = y_score_total.iloc[1:,:]

print('total test set, accuracy =
%.2f'%metrics.accuracy_score(y_true_total, y_pred_total))

print('total test set, AUC = %.2f'%metrics.roc_auc_score(y_true_total,
y_score_total))

    

#%% (7)构建策略 
==等权

para.n_stock_select = long_choose

length = para.month_in_test[-1] - para.month_in_test[0]

strategy = pd.DataFrame({'return':[0] * length,'value':[1] *
length,

                        
'benchmark-r':[0]* length,'benchmark-v':[1]* length})   

a = 0

for i_month in para.month_in_test:    

    y_true_curr_month
= y_true_test.iloc[:,a]

    y_score_curr_month
= y_score_test.iloc[:,a]

    y_score_curr_month
= y_score_curr_month.sort_values(ascending=False)

    index_select =
y_score_curr_month[0:para.n_stock_select].index

   
strategy.loc[a,'return'] = np.mean(y_true_curr_month[index_select])

    strategy['value']
= (strategy['return']+1).cumprod()

   
strategy.loc[a,'benchmark-r'] = np.mean(y_true_curr_month)

   
strategy['benchmark-v'] = 
(strategy['benchmark-r']+1).cumprod()

    a=a+1

    

plt.plot(strategy.index,strategy['value'],'r-')

plt.show()

 

plt.figure(2,figsize=(10, 6)) 

lines1 =
plt.plot(strategy.index,strategy['value'],label='svm')

lines2 =
plt.plot(strategy.index,strategy['benchmark-v'],label='equal index ')

plt.grid(True)  

plt.legend( loc = "upright")

 

ann_excess_return = np.mean(strategy['return']) * 251

ann_excess_vol = np.std(strategy['return']) * np.sqrt(251)

info_ratio = ann_excess_return/ann_excess_vol

 

ann_excess_return2 = np.mean(strategy['benchmark-r']) * 251

ann_excess_vol2 = np.std(strategy['benchmark-r']) *
np.sqrt(251)

info_ratio2 = ann_excess_return/ann_excess_vol

 

print('annual excess return = %.2f'%ann_excess_return)

print('annual excess volatility = %.2f'%ann_excess_vol)

print('information ratio = %.2f'%info_ratio)

 

#%% (8) 改进

'''（8.1）  删除多余相关因子 从结果观测〉0.9

因子4和5强相关；因子6，7，8强相关]

删除5，7，8 保留4流通股本，6市盈率

看下一个 "SVM-50-6因子" 文件 ''' 

 

 

 

'''（8.1）  动态SVM  每次train data 两年，分成0.7和0.3 '''

 

for i in range(start):

    date_index =  df_col_data.index[(-start0-i):(-start-i)]

    data_curr_month =
final_data.iloc[0:1,:]

   
data_curr_month['return_bin'] = np.nan

    

   
SVM_train(final_data,date_index)

    #model

    

 

def SVM_train(final_data,date_index):

    for i_month in
date_index:

       
data_curr_month2 = final_data[(final_data.index == i_month)].dropna(axis
= 0)

       
data_curr_month2 = label_data(data_curr_month2)

       
data_curr_month = pd.concat([data_curr_month,data_curr_month2],axis=0)

    data_in_sample
=  data_curr_month.iloc[1:,:]   

    X_in_sample =
data_in_sample.iloc[:,3:7]     

    y_in_sample =
data_in_sample.iloc[:,-1]

    from
sklearn.model_selection import train_test_split

    X_train,X_cv,y_train,y_cv=train_test_split(X_in_sample,y_in_sample,test_size=para.percent_cv,random_state=para.seed)

    from sklearn
import decomposition #主成分分析

    pca =
decomposition.PCA(n_components = 0.95)

    pca.fit(X_train)

    X_train =
pca.transform(X_train)

    X_cv = pca.transform(X_cv)

    if para.method
=='SVM':

        from sklearn
import svm

        model =
svm.SVC(kernel = para.svm_kernel,C=para.svm_c)

    if para.method ==
'SVM':

       
model.fit(X_train,y_train)

        y_pred_train =
model.predict(X_train)

        y_score_train
= model.decision_function(X_train)

        y_pred_cv =
model.predict(X_cv)

        y_score_cv =
model.decision_function(X_cv)

    model =
svm.SVC(kernel = para.svm_kernel,C=para.svm_c)    

   
model.fit(X_train,y_train)    

    y_pred_train =
model.predict(X_train)

    y_score_train =
model.decision_function(X_train)  

    y_pred_cv =
model.predict(X_cv)  

    y_score_cv =
model.decision_function(X_cv)

 

'''创建三个空数据集y_true_test、y_pred_test 和y_score_test'''

y_true_test = pd.DataFrame([np.nan] *
np.ones((long_choose,1)))

y_true_test.index = choose_list

y_pred_test = pd.DataFrame([np.nan] *
np.ones((long_choose,1)))

y_pred_test.index = choose_list

y_score_test = pd.DataFrame([np.nan] * np.ones((long_choose,1)))

y_score_test.index = choose_list

 

#因子NAN也算停牌，status只是价格停牌

date_index_test =
df_col_data.index[para.month_in_test[0]:para.month_in_test[-1]]

date_index_test = date_index_test[0:-1]

 

for i_month in date_index_test:

    data_curr_month =
final_data[(final_data.index == i_month)]

   
data_curr_month_dropna = data_curr_month.dropna(axis = 0)

    X_curr_month =
data_curr_month_dropna.iloc[:,3:11]

    X_curr_month =
pca.transform(X_curr_month)

    

    y_pred_curr_month
= model.predict(X_curr_month)

    y_score_curr_month
= model.decision_function(X_curr_month)

    

    y_true_df =
data_curr_month['return']

    y_true_df.index =
data_curr_month['简称']

    y_pred_df =
pd.DataFrame(y_pred_curr_month)

    y_pred_df.index = data_curr_month_dropna['简称']

    y_score_df =
pd.DataFrame(y_score_curr_month)

    y_score_df.index =
data_curr_month_dropna['简称']

    

    y_true_test =
pd.concat([y_true_test,y_true_df],axis=1)

    y_pred_test =
pd.concat([y_pred_test,y_pred_df],axis=1)

    y_score_test =
pd.concat([y_score_test,y_score_df],axis=1)

 

y_true_test = y_true_test.iloc[:,1:]

y_pred_test = y_pred_test.iloc[:,1:]

y_score_test = y_score_test.iloc[:,1:]

 

 

 

T= 104 #两年104个数据

 

 

for i_month in date_index:

    data_curr_month2 =
final_data[(final_data.index == i_month)].dropna(axis = 0)

    data_curr_month2 =
label_data(data_curr_month2)

    data_curr_month =
pd.concat([data_curr_month,data_curr_month2],axis=0)

data_in_sample = 
data_curr_month.iloc[1:,:]   

 

X_in_sample = data_in_sample.iloc[:,3:7]     

y_in_sample = data_in_sample.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train,X_cv,y_train,y_cv=train_test_split(X_in_sample,y_in_sample,test_size=para.percent_cv,random_state=para.seed)

 

from sklearn import decomposition #主成分分析

pca = decomposition.PCA(n_components = 0.95)

pca.fit(X_train)

X_train = pca.transform(X_train)

X_cv = pca.transform(X_cv)

 

'''SVM'''

if para.method =='SVM':

    from sklearn
import svm

    model =
svm.SVC(kernel = para.svm_kernel,C=para.svm_c)

if para.method == 'SVM':

   
model.fit(X_train,y_train)

    y_pred_train =
model.predict(X_train)

    y_score_train =
model.decision_function(X_train)

    y_pred_cv =
model.predict(X_cv)

    y_score_cv =
model.decision_function(X_cv)

 

 

 

 

 

 

 

 

 

 

'''（8.3）  组合 '''

 

 

 

 

 

#%%

