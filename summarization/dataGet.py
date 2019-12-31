import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from xml.dom import minidom
import os
import numpy as np
import tensorflow as tf
import re

def preprocess_sentence(w):
  w = w.lower().strip()

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  #w = re.sub(r"([?.!,¿])", r" \1 ", w)
  #w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z,0-9, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z,&\n]+", " ", w)

  w = w.rstrip().strip()

  return w

def read_xml(filename):
    #打开这个文档，用parse方法解析
    xml=minidom.parse(filename)

    #获取根节点
    root=xml.documentElement

    #得到根节点下面所有的book节点
    #更多方法可以参考w2school的内容或者用dir(root)获取
    bugreports = root.getElementsByTagName('BugReport')
    bugslist = []
    #遍历处理，elements是一个列表
    for bugreport in bugreports:
        bugdict = {'BugreportID':0,'Title':'','Content':[]}
        #判断是否有id属性
        if bugreport.hasAttribute('ID'):
            #不加上面的判断也可以，若找不到属性，则返回空
            bugreportId = int(bugreport.getAttribute('ID'))
            bugdict['BugreportID'] = bugreportId
        title = bugreport.getElementsByTagName('Title')[0].firstChild.data
        bugdict['Title'] = title[1:-1]
        turns = bugreport.getElementsByTagName('Turn')
        turnslist = []
        for turn in turns:
            turndict = {'Date':'','Person':'','Text':[]}
            date = turn.getElementsByTagName('Date')[0].firstChild.data
            person = turn.getElementsByTagName('From')[0].firstChild.data
            textNode = turn.getElementsByTagName('Sentence')
            turndict['Date'] = date[1:-1]
            turndict['Person'] = person[1:-1]
            textlist = []
            for i,sentenceNode in enumerate(textNode):
                sentence_id = sentenceNode.getAttribute('ID')
                sentence_context = sentenceNode.firstChild.data
                sentence_tuple = (sentence_id, sentence_context)
                textlist.append(sentence_tuple)
            turndict['Text'] = textlist
            turnslist.append(turndict)
        bugdict['Content'] = turnslist
        bugslist.append(bugdict)
    return bugslist

def read_label(filename):
    label = []
    with open(filename, 'r') as file:
        for line in file:  # 设置文件对象并读取每一行文件
            line = line.strip().split(',')
            label.append(line)
    print(label)
    num = int(label[-1][0])
    print('num',num)
    labellist = []
    for i in range(num):
        list = []
        for index,id in label:
            if int(index)==i+1:
                list.append(str(id))
        labellist.append(list)

    return labellist

def get_Content(bugslist):
    samples = []
    ids = []
    for bugreport in bugslist:
        titles = []
        id = []
        sentences = []
        titles.append(bugreport['Title'])
        for turn in bugreport['Content']:
            for id_sentence, sentence in turn['Text']:
                sentences.append(sentence)
                id.append(str(id_sentence))
        ids.append(id)
        samples.append(sentences)

    return samples,ids

def savefile(samples):
    num_sentence = []
    len_sentence = []
    for i,sample in enumerate(samples):
        num = 0
        string = ''

        for sentence in sample:
            string = string + sentence + '\n'
            num = num+1

        num_sentence.append(num)
        with open("./bug report/" + str(i+1)+".txt", "w", encoding='utf-8') as f:
            f.write(str(string))
    return num_sentence

def num_word(samples):
    num_word_list = []
    for sample in samples:
        list = []
        for sentence in sample:
            list.append(len(sentence.split(' ')))
        num_word_list.append(list)
    numword = [sum(i) for i in num_word_list]
    return num_word_list,numword

def textrank(text,sum_word,len_sentence):
    tr4s = TextRank4Sentence(delimiters='\n')
    tr4s.analyze(text=text, lower=True, source='all_filters')
    print('摘要：')
    sentences = []
    weight = []
    index = []

    for item in tr4s.get_key_sentences(num=1000):
        print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重
        sentences.append(item.sentence)
        weight.append(item.weight)
        index.append(item.index)
    extra_word = int(sum_word*0.32)
    num = 0

    extra_num = 0
    for data in len_sentence:
        if num+data<extra_word:
            num = num+data
            extra_num = extra_num + 1
        else:
            break

    result = index[:extra_num]

    return result

def bugsum(path,sum_word,len_sentence):

    result_samples = []
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4])) #去掉文件扩展名进行排序，即去掉'.txt'
    print(files)
    for i,file in enumerate(files):

        file_path = os.path.join(path, file)
        print('路径', file_path)
        text = codecs.open(file_path, 'r', 'utf-8').read()
        result = textrank(text,sum_word[i],len_sentence[i])
        result_samples.append(result)
    return result_samples

def index2id(indexlist,idlist):
    extra_ids = []
    for i,j in zip(indexlist,idlist):
        index = np.array(i)
        print('index',index)
        id = np.array(j)
        print('id',id)
        extra_id= list(id[index])
        print('extra_id',extra_id)
        extra_ids.append(extra_id)
    return extra_ids

def evaluate(y,pred):

    acclist = []
    prlist = []
    relist = []
    f1list = []
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    accuracy = tf.keras.metrics.Accuracy()
    for per_y,per_pred in zip(y,pred):
        #计算recall
        recall.update_state(per_y, per_pred)
        re = recall.result()
        print('recall',re)
        #计算precision
        precision.update_state(per_y, per_pred)
        pr = precision.result()
        print('precision',pr)
        #计算f1
        if pr+re == 0.:
            f1 = 0
        else:
            f1 = 2.0*pr*re/(pr+re)
        print('f1',f1)
        #计算accuracy
        accuracy.update_state(per_y, per_pred)
        acc = accuracy.result()
        print('acc',acc)
        prlist.append(pr)
        relist.append(re)
        acclist.append(acc)
        f1list.append(f1)

    mean_pr = np.mean(prlist)
    mean_re = np.mean(relist)
    mean_acc = np.mean(acclist)
    mean_f1 = np.mean(f1list)
    return mean_acc, mean_pr, mean_re, mean_f1

def index2pred(index,ids):
    index = np.array(index)
    idlist = [np.zeros_like(id,dtype=np.int32) for id in ids]
    for i, id in enumerate(idlist):
        id[index[i]] = 1
    idlist = [list(id) for id in idlist]
    return idlist

def label2y(label,ids):
    y = ids.copy()
    for index,(id,la) in enumerate(zip(y,label)):
        #ids[index][0] = len(id)
        for i,value in enumerate(id):
            if value in la:
                y[index][i] = 1
            else:
                y[index][i] = 0
    return y

def main():
    path = './bug report'
    bugslist = read_xml('bugreports.xml')
    print(bugslist)
    label = read_label('goldset.txt')
    print(label)
    samples,ids = get_Content(bugslist)
    print('ids',ids)
    #num_sentence = [len(i) for i in ids]
    num_word_list,numword = num_word(samples)
    num_sentence = savefile(samples)
    print(numword,num_word_list)
    result = bugsum(path,numword,num_word_list)
    print(result)
    extra_ids = index2id(result,ids)
    print(extra_ids)
    pred = index2pred(result,ids)
    print('pred',pred)
    y = label2y(label,ids)
    print('y',y)
    mean_acc, mean_pr, mean_re, mean_f1 = evaluate(y,pred)
    print('mean_acc, mean_pr, mean_re, mean_f1',mean_acc, mean_pr, mean_re, mean_f1)

if __name__ == '__main__':
    main()
