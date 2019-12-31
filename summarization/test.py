import numpy as np
import tensorflow as tf
def evaluate(y,pred):

    mean_acc = 0
    mean_pr = 0
    mean_re = 0
    mean_f1 = 0
    for batch_y,batch_pred in zip(y,pred):
        label_num = tf.cast(batch_y[0],dtype=tf.int32)
        y_true = batch_y[1:]
        print('pred',batch_pred[:label_num])
        print('y',y_true[:label_num])
        b_pred = batch_pred[:label_num]
        b_y = y_true[:label_num]
        #计算recall
        recall = tf.keras.metrics.Recall()
        recall.update_state(b_y, b_pred)
        re = recall.result()
        print('recall',re)
        #计算precision
        precision = tf.keras.metrics.Precision()
        precision.update_state(b_y, b_pred)
        pr = precision.result()
        print('precision',pr)
        #计算f1
        if pr+re == 0.:
            f1 = 0
        else:
            f1 = 2.0*pr*re/(pr+re)
        print('f1',f1)
        #计算accuracy
        accuracy = tf.keras.metrics.Accuracy()
        accuracy.update_state(b_y, b_pred)
        acc = accuracy.result()
        print('acc',acc)
        '''
        correct = tf.equal(batch_pred[:label_num], y_true[:label_num])
        correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32), axis=-1)/label_num
        print('correct',correct)
        '''
        mean_pr = mean_pr + pr
        mean_re = mean_re + re
        mean_acc = mean_acc + acc
        mean_f1 = mean_f1 + f1

    mean_pr = mean_pr / len(y)
    mean_pr = tf.cast(mean_pr, dtype=tf.float32)
    mean_re = mean_re / len(y)
    mean_re = tf.cast(mean_re, dtype=tf.float32)
    mean_acc = mean_acc / len(y)
    mean_acc = tf.cast(mean_acc, dtype=tf.float32)
    mean_f1 = mean_f1 / len(y)
    mean_f1= tf.cast(mean_f1, dtype=tf.float32)
    print(mean_f1)
    return mean_acc, mean_pr, mean_re, mean_f1

#y =tf.cast([1,0,1],dtype=tf.float32)
#pred = tf.cast([0,0,1],dtype=tf.float32)
ids = [['1','2'],['3','4','5']]
index = np.array([[1],[0]])
idlist = [np.zeros_like(id,dtype=np.int32) for id in ids]
for i,id in enumerate(idlist):
    id[index[i]] = 1
print(idlist)


y = [[1,0,1]]
pred = [[0,1,1]]
recall = tf.keras.metrics.Recall()
recall.update_state(y, pred)
re = recall.result()
print('recall',re)