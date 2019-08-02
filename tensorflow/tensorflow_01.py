# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:04:37 2019

@author: 002954
"""

import time
import numpy as np
import tensorflow as tf
import random
from collections import Counter
import pymysql
import pandas as pd


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
    word_result= execude_sql("""SELECT  b.word from spider_36kr_record as a JOIN  news_word_relation as b ON a.id=b.new_id""")
    if word_result is None:
        return words
    
    if word_result.columns.size<=0:
        return words
            
        
   
    words=list(word_result[0])
  
    
    return words      






print("获取关键字开始")
start = time.time()
words=get_words()
end = time.time()
print("获取关键字结束 cost ",((end-start)/100))
len(words)
vocab = set(words)
vocab_to_int = {w: c for c, w in enumerate(vocab)}
int_to_vocab = {c: w for c, w in enumerate(vocab)}

print("total words: {}".format(len(words)))
print("unique words: {}".format(len(set(words))))
# 对原文本进行vocab到int的转换
int_words = [vocab_to_int[w] for w in words]



t = 1e-5 # t值
threshold = 0.9 # 剔除概率阈值

# 统计单词出现频次
int_word_counts = Counter(int_words)
total_count = len(int_words)
# 计算单词频率
word_freqs = {w: c/total_count for w, c in int_word_counts.items()}
# 计算被删除的概率
prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
# 对单词进行采样
train_words = [w for w in int_words if prob_drop[w] > threshold]
#print("threshold ",threshold,"prob_drop ",prob_drop,"train_words ",len(train_words))
print("train_words ",len(train_words))








# 输入
train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
    
vocab_size = len(int_to_vocab)
embedding_size = 200 # 嵌入维度
with train_graph.as_default():
    # 嵌入层权重矩阵
    embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    # 实现lookup
    embed = tf.nn.embedding_lookup(embedding, inputs)


# Negative Sampling
n_sampled = 100
with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(vocab_size))
    
    # 计算negative sampling下的损失
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)
    
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)



#验证
with train_graph.as_default():
    # 随机挑选一些单词
    ## From Thushan Ganegedara's implementation
    valid_size = 7 # Random set of words to evaluate similarity on.
    valid_window = 100
    # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, 
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = [vocab_to_int['大会'], 
                      vocab_to_int['5月'], 
                      vocab_to_int['网络'],
                      vocab_to_int['医疗'], 
                      vocab_to_int['平台'], 
                      vocab_to_int['综合性'],
                      vocab_to_int['比特币']]
    
    valid_size = len(valid_examples)
    # 验证单词集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # 计算每个词向量的模并进行单位化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    # 查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # 计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))



def get_batches(words, batch_size, window_size=5):
    '''
    构造一个获取batch的生成器
    '''
    n_batches = len(words) 
    # 仅取full batches
    words = words[:n_batches*batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx: idx+batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(batch, i, window_size)
            # 由于一个input word会对应多个output word，因此需要长度统一
            x.extend([batch_x]*len(batch_y))
            y.extend(batch_y)
        yield x, y





def get_targets(words, idx, window_size=5):
    '''
    获得input word的上下文单词列表
    
    参数
    ---
    words: 单词列表
    idx: input word的索引号
    window_size: 窗口大小
    '''
    target_window = np.random.randint(1, window_size+1)
    # 这里要考虑input word前面单词不够的情况
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    # output words(即窗口中的上下文单词)
    targets = set(words[start_point: idx] + words[idx+1: end_point+1])
    return list(targets)


                
        

epochs = 10 # 迭代轮数
batch_size = 1000 # batch大小
window_size = 10 # 窗口大小

with train_graph.as_default():
    saver = tf.train.Saver() # 文件存储

with tf.Session(graph=train_graph) as sess:
    print("begin")
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs+1):
        batches = get_batches(train_words, batch_size, window_size)
      
        start = time.time()
         
        for x, y in batches:
           
            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            
            loss += train_loss
            if iteration % 10 == 0: 
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss/100),
                      "{:.4f} sec/batch".format((end-start)/100))
                loss = 0
                start = time.time()
            
            # 计算相似的词
            if iteration % 100 == 0:
                # 计算similarity
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 1 # 取最相似单词的前8个
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
            
            iteration += 1
            
    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)


