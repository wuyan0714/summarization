import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import numpy as np

text = codecs.open('./article.txt', 'r', 'utf-8').read()
'''
tr4w = TextRank4Keyword()

tr4w.analyze(text=text, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

print( '关键词：' )
for item in tr4w.get_keywords(20, word_min_len=1):
    print(item.word, item.weight)

print()
print( '关键短语：' )
for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num= 2):
    print(phrase)
'''

tr4s = TextRank4Sentence(delimiters='\n')
tr4s.analyze(text=text, lower=True, source='all_filters')
print('摘要：')
sentences = []
weight = []
index = []

for item in tr4s.get_key_sentences(num=50):
    print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重
    sentences.append(item.sentence)
    weight.append(item.weight)
    index.append(item.index)
