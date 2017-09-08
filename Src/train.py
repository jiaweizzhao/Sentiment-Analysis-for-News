#!/usr/bin/env python
#神经网络

import keras
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import word2vec #用于词向量表的读取
#相关程序
import prePocess
#系统路径调用
import sys
import os

#相关参数
VALIDATION_SPLIT = 0.2 #验证集占比
EMBEDDING_DIM = 100    #EMBEDDING维度
MAX_SEQUENCE_LENGTH = 93  #Seq最大维度


#__main__

projectPath = os.path.abspath('').rstrip('/Sentiment-Analysis-for-News/Src') + 'News'

label = []
example = []
prePocess.loadText(example,label,projectPath)

#模型构建

#Text to Seq
tokenizer = Tokenizer()  #tokenizer用于sequence的一些处理，由文本到seq
tokenizer.fit_on_texts(example)
sequences = tokenizer.texts_to_sequences(example) #这一步将文本词汇使用one-hot变为int，并且根据频率创建了一个词典，用于index和文本之间的查询

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = sequence.pad_sequences(sequences)

label = keras.utils.to_categorical(label,2) #因为Keras只支持单类的01表示，所以一个类问题要变为两个类有无的问题
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', label.shape)

##split the data into a training set and a validation set

##将样本随机化
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
label = label[indices]

##分割训练与验证样本
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = label[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = label[-nb_validation_samples:]

#Embedding layer设置
'''解释：
		目前Embedding layer的这个矩阵只是包含训练集中的词语，故矩阵的维度较小，这样便于训练。
		在新闻测试时可以将维度开放到所有Word2vec上，这样可能取到更好的效果
   要把前面sequence得到的单词向量以及word2vec的向量进行关联
'''

embeddings_index = {}

model = word2vec.load(projectPath +"/Data_final/word2vec/corpusWord2Vec.bin")  

##统计数据 保留接口
numWords,numVectors = model.vectors.shape

print('Found %s word vectors.' % numWords)

#建立词向量矩阵
##首先创建一个100维度的0向量
zero_100 = []
for i in range(100):
	zero_100.append(0)
##正式建立
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    try:
    	embedding_vector = model[word]
     	# words not found in embedding index will be all-zeros.
    except KeyError as e:
       	embedding_matrix[i] = zero_100
       	pass
    else:
       	embedding_matrix[i] = embedding_vector

#与keras Embedding矩阵关联 放在了后面的创建中


#神经网络  其他层设置   使用LSTM与CNN结合方式

model = Sequential()
model.add(Embedding(len(word_index) + 1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False))
model.add(Conv1D(filters=100, kernel_size=10, padding='same', activation='relu'))#
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100,dropout=0.2, recurrent_dropout=0.2)) #加入Dropout
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#训练及误差结果
model.fit(x_train, y_train, epochs=1, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(x_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#保存模型
model.save("Model with dropout")












