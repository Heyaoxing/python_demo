#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tushare as ts
import pymysql
import pandas as pd
import time
import numpy as np
import datetime


# In[8]:


def batch_install_sql(sql,data):
# 创建连接
    try:
        db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='stock', charset='utf8mb4')
    except:
        print('数据库连接失败，10s后重试')
        time.sleep(10)
# 创建游标
    cursor = db.cursor()
    cursor.executemany(sql,data)
# 涉及写操作要注意提交
    db.commit()
# 关闭连接
    db.close()    
    cursor.close()


# In[9]:


def install_data(df_data,code):
    sql='INSERT INTO stock.tick_data_'+code+'(time, price,  volume, amount, type, changes, code,create_tm) VALUES (%s,%s,%s,%s,%s,%s,'+code+',now())'
    batch_install_sql(sql,df_data)


# In[10]:


def datadframe_page(df,page,limit):
    return df[(int(page) - 1) * int(limit): (int(page) * int(limit))]


# In[11]:


def get_stock_data(code,date):
    df = ts.get_tick_data(code,date=date,src='tt')
    if df is None:
        print(code+' '+date+' is none')
        return
    
    df['time']=date+' '+df['time']
    print(len(df))
    page=1
    limit=2000

    while (page-1)*limit < len(df):
        df_page=datadframe_page(df,page,limit)
        install_data(np.array(df_page[['time', 'price',  'volume', 'amount', 'type','change']]).tolist(),code)
        page=page+1
        print(page)


# In[12]:



code='600848'
days=0
while days>-10:
    days=days-1
    date=datetime.timedelta(days=days)+datetime.datetime.now()
    date_str=date.strftime('%Y-%m-%d')
    print(date_str)
    get_stock_data(code,date_str)
  


# In[32]:


def get_all(day):
    codes=ts.get_stock_basics().index.tolist() 
    for code in codes:
        print('process code '+code)


# In[ ]:


def get_stock_provcess(code,day):
    index=0
    while index>-day:
        index=index-1
        date=datetime.timedelta(days=index)+datetime.datetime.now()
        date_str=date.strftime('%Y-%m-%d')
        print(date_str)
        get_stock_data(code,date_str)
        
 # processed 002966,688002,688009,300127,603327,603613,300748,603233,603687,300526,603279,002881,600371,300223,002129,300786,002957,002201,300119,603882,000687,000831,603256,600980,000590,600366,688010,688003,300224,688028,600111,300174,000713,000795,002057,600259,600206,600483,688020,000970,688033,603258,002056,300328,000758,002378,600313,002876,300777,603712,600195,002041,300378,688005,688011,600392,600354,300779,002842


 get_all(3650)
