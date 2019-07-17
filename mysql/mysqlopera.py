# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:40:13 2019

@author: 002954
"""

import pymysql
import pandas as pd
import time


def execude_sql(sql):
# 创建连接
    try:
        db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='jeecg', charset='utf8mb4')
    except:
        print('数据库连接失败，10s后重试')
        time.sleep(10)
# 创建游标
    cursor = db.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
#执行结果转化为dataframe
    df = pd.DataFrame(list(result))
# 关闭连接
    db.close()
#返回dataframe
    
    return df



def get_words():
    words=[]
    word_result= execude_sql("""SELECT  b.word from spider_36kr_record as a JOIN  news_word_relation as b ON a.id=b.new_id   LIMIT 10""")
    if word_result is None:
        return words
    
    if word_result.columns.size<=0:
        return words
            
        
   
    words=list(word_result[0])
  
    
    return words    

print(get_words())