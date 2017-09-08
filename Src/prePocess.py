#!/usr/bin/env python

#xml文件读取
from xml.dom.minidom import parse
import xml.dom.minidom
#用于分词
import jieba  
import numpy


#文本读取  使用前修改读取位置
def loadText(example,label,projectPath):
	#Weibo
	for numFile in range(20) :
		with xml.dom.minidom.parse(projectPath + "/Data_final/Traning Data/Weibo/" + str(numFile+1) + ".xml") as DOMTree:
			Result = DOMTree.documentElement
			sentences = Result.getElementsByTagName("sentence")

			for sentence in sentences:
				if sentence.hasAttribute("polarity"):

					#判断输出情感是否为其他值

					sentence_temp1 = sentence.childNodes[0].data.replace('#', '') 
					sentence_temp2 = [' '.join(list(jieba.cut(sentence_temp1,cut_all=False)))]

					#去除分词残留的'' 不好去除，在后面embblding阶段不去表示它

					#判断输出情感是否为其他值,不是的话即可加入训练集
					if sentence.getAttribute("polarity") != 'NEU':
						if sentence.getAttribute("polarity") != 'OTHER':
							example.append(sentence_temp2[0])
							label.append(sentence.getAttribute("polarity"))

					#测试
					#print(sentence_temp2[0])
					#print(sentence_temp2[3])
					#print(sentence.getAttribute("polarity"))
	#Camera&Phone
	for numFile in range(4) :
		with xml.dom.minidom.parse(projectPath + "/Data_final/Traning Data/Camera/" + str(numFile+1) + ".xml") as DOMTree:
			Reviews = DOMTree.documentElement
			Review_all = Reviews.getElementsByTagName("Review")

			for Review in Review_all:
				sentences = Review.getElementsByTagName("sentences")
				sentence_all = sentences[0].getElementsByTagName("sentence")

				for sentence in sentence_all:
					Opinions = sentence.getElementsByTagName("Opinions")
					if Opinions != []:
						
						text = sentence.getElementsByTagName("text")
						sentence_temp1 = text[0].childNodes[0].data.replace('#', '') 
						sentence_temp2 = [' '.join(list(jieba.cut(sentence_temp1,cut_all=False)))]

						example.append(sentence_temp2[0])


						Opinion = Opinions[0].getElementsByTagName("Opinion")				
						label.append(Opinion[0].getAttribute("polarity"))

						#测试
						#print(sentence_temp2[0])
						#print(Opinion[0].getAttribute("polarity"))
	#数据最后处理
	#统一label
	for index in range(len(label)):
		if label[index] == 'negative':
			label[index] = 'NEG'
		if label[index] == 'positive':
			label[index] = 'POS'

	#将字符变为数字  后面的label = keras.utils.to_categorical(label,2)无法使用，故手动转化
	for index in range(len(label)):
		if label[index] == 'NEG':
			label[index] = 0
		if label[index] == 'POS':
			label[index] = 1
