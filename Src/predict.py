#!/usr/bin/env python

import keras
import pickle
import jieba
import os

def predict(text):

	#文本预处理
	projectPath = os.path.abspath('').rstrip('/Sentiment-Analysis-for-News/Src') + 'News'

	tokenizer = pickle.load(open(projectPath + "/Data_final/tokenizerFile", 'rb'))

	text = [' '.join(list(jieba.cut(text,cut_all=False)))]

	sequence = tokenizer.texts_to_sequences(text)

	data = keras.preprocessing.sequence.pad_sequences(sequence)

	#模型预测

	model = keras.models.load_model(projectPath + "/Sentiment-Analysis-for-News/Src/Model with dropout")

	#keras.utils.plot_model(model,to_file = projectPath + "/Sentiment-Analysis-for-News/Src/model.png")

	return model.predict(data,2)


text = "我好喜欢你呀"
result = predict(text)
print(result)



