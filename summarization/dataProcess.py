import re
import tensorflow as tf
import numpy as np
from dataGet import read_xml,read_label
def preprocess_sentence(w):
  w = w.lower().strip()

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z,0-9, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿0-9]+", " ", w)

  w = w.rstrip().strip()

  return w

def tokenize(dictlist):
    #sentence = "(495584) Firefox - search suggestions a passes wrong previous result to form history"
    #sentence = preprocess_sentence()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(dictlist)

    tensor = tokenizer.texts_to_sequences(dictlist)
    tokenizer.sequences_to_texts(tensor)
    #print(tokenizer.sequences_to_texts(tensor))
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    dictlen = np.max(tensor)
    return dictlen,tokenizer



def getContent(bugslist):
    title_list = []
    sentence_list = []
    id_sentence_list = []
    for bugreport in bugslist:
        #persons = []
        sentenceslist = []
        idslist = []
        title_list.append(bugreport['Title'])
        for turn in bugreport['Content']:
            #persons.append(turn['Person'])
            sentences = []
            ids = []
            for id_sentence, sentence in turn['Text']:
                sentences.append(sentence)
                ids.append(id_sentence)
            sentenceslist.append(sentences)
            idslist.append(ids)
        sentence_list.append(sentenceslist)
        id_sentence_list.append(idslist)
        print(sentence_list)
        print(idslist)
    return title_list,sentence_list,id_sentence_list

def getDict(bugslist):
    title_list = []
    sentence_list = []
    id_sentence_list = []
    for bugreport in bugslist:
        title_list.append(bugreport['Title'])
        for turn in bugreport['Content']:
            for id_sentence, sentence in turn['Text']:
                sentence_list.append(sentence)
                id_sentence_list.append(id_sentence)
    total_list = title_list+sentence_list
    return total_list,title_list,sentence_list,id_sentence_list

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
        print(sentences)
        sample = titles+sentences
        samples.append(sample)
    #print('ids',ids)
    return samples,ids

def dataprocess(bugfilename,labelfilename):
    bugslist = read_xml(bugfilename)
    label = read_label(labelfilename)
    total_list, title_list, sentence_list, id_sentence_list = getDict(bugslist)
    dictlen, tokenizer = tokenize(total_list)
    samples ,ids= get_Content(bugslist)  # 第0位为title，之后为sentence
    samples_token = []
    for sample in samples:
        sample_token = tokenizer.texts_to_sequences(sample)
        sample_token = tf.keras.preprocessing.sequence.pad_sequences(sample_token, padding='post',value = -1)
        samples_token.append(sample_token)
    samples_token,ids_label = pad_sentence(samples_token,ids,label)
    #print(tokenizer.sequences_to_texts(samples_token[0]))
    #samples_token = tf.cast(samples_token, dtype=tf.float32)
    ids_label = tf.cast(ids_label,dtype=tf.float32)
    return samples_token,ids_label,dictlen

def pad_sentence(samples_token,ids,label):
    rowmaxlen = np.max([len(sample_token) for sample_token in samples_token])
    colmaxlen = np.max([len(sample_token[0]) for sample_token in samples_token])
    idmaxlen = np.max([len(id) for id in ids]) #idmaxlen = rowmaxlen-1
    for index,(id,la) in enumerate(zip(ids,label)):
        #ids[index][0] = len(id)
        for i,value in enumerate(id):
            if value in la:
                ids[index][i] = 1
            else:
                ids[index][i] = 0
        ids[index].insert(0,len(id))
    for i,id in enumerate(ids):
        ids[i] =np.pad(id,(0,idmaxlen+1-len(id)))
    for i,sample_token in enumerate(samples_token):
        samples_token[i] = np.pad(sample_token, ((0,rowmaxlen-len(sample_token)),
            (0,colmaxlen-len(sample_token[0]))),'constant', constant_values=(-1,-1))
    return samples_token,ids

if __name__ == '__main__':
    samples_token,ids_label,dictlen= dataprocess('bugreports.xml','goldset.txt')
    print(samples_token[0])
    print('ids',tf.reduce_sum(ids_label[23])-ids_label[23][0])
