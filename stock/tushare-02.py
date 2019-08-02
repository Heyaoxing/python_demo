#!/usr/bin/env python
# coding: utf-8

# In[16]:


import tushare as ts
import pymysql
import pandas as pd
import time
import numpy as np
import datetime


# In[17]:


def execude_sql(sql):
# 创建连接
    try:
        db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='stock', charset='utf8mb4')
    except:
        print('数据库连接失败，10s后重试')
        time.sleep(10)
# 创建游标
    cursor = db.cursor()
    cursor.execute(sql)
#执行结果转化为dataframe
# 关闭连接
    cursor.close()
    db.close()


# In[ ]:


codes=ts.get_stock_basics().index.tolist() 
for code in codes:
    sql='create table if not exists tick_data_'+code+' like tick_data'
    print(sql)
    execude_sql(sql)
       
        


# In[ ]:




