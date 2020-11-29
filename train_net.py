import tensorflow as tf
import numpy as np
import os 
import cnn_inference
import time_window
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn import preprocessing

from sklearn.metrics import roc_curve, auc  ###计算roc和auc




def acu_curve(y,prob):
#y真实prob预测
    fpr,tpr,threshold = roc_curve(y,prob) ###计算真阳性率和假阳性率
    roc_auc = auc(fpr,tpr) ###计算auc的值
 
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
 
    plt.show()




'''首先配值训练一个神经网络可能需要用到的各种参数'''
list1 = [(n+1) for n in range(190)]
RMSE_RECORD=[]
tpr_record=[]
fpr_record=[]


BATCH_SIZE=128


LEARNING_RATE_BASE=0.9
LEARNING_RATE_DECAY=0.9 
REGULARAZTION_RATE=0.01
DIM=26
TRAINING_STEPS=66000
tf.reset_default_graph()
'''需要将训练好的模型的参数进行保存'''
MODEL_SAVE_PATH="/path/to/model"
MODEL_NAME="model.ckpt"
'''准备好要输入到模型，去测试模型的初始数据'''
#X_train,X_test,Y_train,Y_test,no_X_test,no_Y_test,no_X_train,no_Y_train,no_X_train_down,no_X_test_down,no_Y_train_down,no_Y_test_down=time_window.data_perprocessing()
#no_X_test,no_Y_test,no_X_train,no_Y_train,no_X_train_down,no_X_test_down,no_Y_train_down,no_Y_test_down,X_train_21,X_test_21,Y_train_21,Y_test_21=time_window.data_perprocessing()
X_train,X_test,Y_train,Y_test=time_window.data_perprocessing()
'''X_train,X_test,Y_train,Y_test,no_X_train,no_Y_train,no_X_test,no_Y_test=time_window.data_perprocessing()'''
'''定义好输入和输出'''




#train_num=len(X_train)
#X_train=np.reshape(X_train,(train_num,48,DIM,1))
#test_size=len(X_test)
#X_test=np.reshape(X_test,(test_size,48,DIM,1))

'''
no_test_size=len(no_X_test)
no_X_test=np.reshape(no_X_test,(no_test_size,48,DIM,1))
no_train_size=len(no_X_train)
no_train_num=no_train_size
no_X_train=np.reshape(no_X_train,(no_train_size,48,DIM,1))

no_test_size_down=len(no_X_test_down)
no_X_test_down=np.reshape(no_X_test_down,(no_test_size_down,48,DIM,1))
no_train_size_down=len(no_X_train_down)
no_train_num_down=no_train_size_down
no_X_train_down=np.reshape(no_X_train_down,(no_train_size_down,48,DIM,1))

test_size_21=len(X_test_21)
X_test_21=np.reshape(X_test_21,(test_size_21,48,DIM,1))
train_size_21=len(X_train_21)
train_num_21=train_size_21
X_train_21=np.reshape(X_train_21,(train_size_21,48,DIM,1))
'''
test_size=len(X_test)
X_test=np.reshape(X_test,(test_size,106,DIM,1))
train_size=len(X_train)
X_train=np.reshape(X_train,(train_size,106,DIM,1))












'''BBB=np.zeros((100,24,26,1))
AAA=np.zeros((100,1))
num=0
for i in range(test_size):
    if Y_test[i]==1:
        BBB[num]=X_test[i]
        AAA[num]=1
        num=num+1
    if num==100:
        break
print(num)'''
x=tf.placeholder(tf.float32,[BATCH_SIZE,106,DIM,1],name='x_input')
y_=tf.placeholder(tf.float32,[BATCH_SIZE,2],name='y_output')

regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

'''用来定义输入和输出的palceholder'''
global_step=tf.Variable(0,trainable=False)
train=1
'''reuse=False'''
y=cnn_inference.inference(x,train,regularizer)
'''reuse=True'''
global_step=tf.Variable(0,trainable=False)
'''定义损失函数  还是得在这个函数上下功夫啊'''
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
#cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y, name=None)
#cross_entropy=tf.nn.weighted_cross_entropy_with_logits(targets=y_,logits=y,pos_weight=1,name=None)
#condition=tf.argmax(y_,1)
#cross_entropy_1= tf.where(condition == 1, cross_entropy, cross_entropy)

cross_entropy_mean=tf.reduce_mean(cross_entropy)

loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))

'''需要进行调整'''
'''计算分类结果的准确率'''
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


predictionss = tf.argmax(y, 1)
actualss = tf.argmax(y_, 1)















'''accuracy1,recall,precision,f1_score=results.tf_results(y,y_)
accuracy1=accuracy1
recall=recall
precision=precision
f1_score=f1_score'''
'''accuracy1=results.tf_results(y,y_)'''
'''recall=results.tf_results(y,y_)'''
'''对学习率进行调整的一种方法'''
'''
learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        no_train_size/BATCH_SIZE,
        LEARNING_RATE_DECAY)
'''


''' 关键一步 反向训练对网络的参数进行更新'''
#train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
#tf.train.AdamOptimizer.__init__(learning_rate=0.0005)
aa=tf.train.AdamOptimizer(learning_rate=0.0005)
train_step=aa.minimize(loss,global_step=global_step)

''' 保存训练过程当中所得到的参数'''
#train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)


'''saver=tf.train.Saver()'''
'''保存训练过程当中的参数'''

'''正式的去开始训练模型'''
with tf.Session()as sess:
    
    tf.global_variables_initializer().run()
    
    #print('训练集的总的样本个数',no_train_num)
    #epoch=int(no_train_num/BATCH_SIZE)
    num_epoch=[]
    loss_record=[]
    loss_test_record=[]
    test_accuracy_record=[]
    for i in range(TRAINING_STEPS):
        '''获得batch输入 xs 和 ys'''
        train=1
        aa=np.random.randint(0,train_size, (BATCH_SIZE)) 
        xs=X_train[aa]
        
        ys=Y_train[aa]
        _,loss_value,step=sess.run([train_step,loss,global_step],feed_dict={x:xs,y_:ys})
        predictions_train=sess.run(predictionss,feed_dict={x:xs,y_:ys})
        actuals_train=sess.run(actualss,feed_dict={x:xs,y_:ys})
        
        
        
        
        
        '''每训练1000轮，计算一次损失函数的值，并把当前所获得的模型的参数进行保存'''
        if (i%1000)==0:
            print("After %d training steps ,loss on training batch is %g" %(int(i/1000),loss_value))
            bb=np.random.randint(0,test_size, (BATCH_SIZE)) 
            xs_test=X_test[bb]
        
            ys_test=Y_test[bb]
            
            loss_test=sess.run([loss],feed_dict={x:xs_test,y_:ys_test})   
            predictions_train=sess.run(predictionss,feed_dict={x:xs_test,y_:ys_test})
            actuals_train=sess.run(actualss,feed_dict={x:xs_test,y_:ys_test})
            TP_number_train=0
            FN_number_train=0
            FP_number_train=0
            TN_number_train=0
        
            for i_train in range(BATCH_SIZE):
                if predictions_train[i_train]==1:
                    if actuals_train[i_train]==1:
                        TP_number_train=TP_number_train+1
                    if actuals_train[i_train]==0:
                        FP_number_train=FP_number_train+1
                if predictions_train[i_train]==0:
                    if actuals_train[i_train]==1:
                        FN_number_train=FN_number_train+1
                    if actuals_train[i_train]==0:
                        TN_number_train=TN_number_train+1
            Accuracy_test=(TP_number_train+TN_number_train)/(TP_number_train+TN_number_train+FP_number_train+FN_number_train+0.000001)
            
            print("在测试集上的损失是     ",loss_test)
            print("After %d training steps ,Accuracy on test batch is %g" %(int(i/1000),Accuracy_test))
            
            num_epoch.append(int(i/1000))
            loss_record.append(loss_value)
            loss_test_record.append(loss_test)
            test_accuracy_record.append(Accuracy_test)
    '''需要对测试集的数据进行一定的处理'''
    RMSE_NUM_NUM=0
            
    
    
    
    
    
    
    
    
    
    
    fprs=0
    tprs=0
    TP_number_num=0
    FP_number_num=0
    FN_number_num=0
    TN_number_num=0
    
    
    
    
    for j in range(190):
        print('sssssssssssssssssssssssssss',X_test.shape)
        test_feed={x:X_test[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                   y_:Y_test[j*BATCH_SIZE:(j+1)*BATCH_SIZE]}
        train=0 
        predictions=sess.run(predictionss,feed_dict=test_feed)
        actuals=sess.run(actualss,feed_dict=test_feed)
        if j==0:
            actuals_num=actuals
            predictions_num=predictions
        actuals_num=np.concatenate([actuals_num,actuals],axis=0)
        predictions_num=np.concatenate([predictions_num,predictions],axis=0)
        
        '''acc1,recall,precision,f1=sess.run(accuracy1,feed_dict=test_feed)'''
        '''acc1=sess.run(accuracy1,feed_dict=test_feed)'''
        '''recall=sess.run(recall,feed_dict=test_feed)'''
        print(predictions[0:10])
        print(actuals[0:10])
        print(predictions[10:20])
        print(actuals[10:20])
        print(predictions[20:30])
        print(actuals[20:30])
    
        print(predictions[30:40])
        print(actuals[30:40])
        print(predictions[40:50])
        print(actuals[40:50])
        print(predictions[50:60])
        print(actuals[50:60])
        print(predictions[60:70])
        print(actuals[60:70])
        print(predictions[70:80])
        print(actuals[70:80])
        print(predictions[80:90])
        print(actuals[80:90])
        print(predictions[90:100])
        print(actuals[90:100])
        print(predictions[100:110])
        print(actuals[100:110])
        print(predictions[110:120])
        print(actuals[110:120])
        print(predictions[120:127])
        print(actuals[120:127])
        
        numbers=actuals.size
        TP_number=0
        FN_number=0
        FP_number=0
        TN_number=0
        
        for i in range(BATCH_SIZE):
            if predictions[i]==1:
                if actuals[i]==1:
                    TP_number=TP_number+1
                if actuals[i]==0:
                    FP_number=FP_number+1
            if predictions[i]==0:
                if actuals[i]==1:
                    FN_number=FN_number+1
                if actuals[i]==0:
                    TN_number=TN_number+1
        TP_number_num=TP_number_num+TP_number
        FP_number_num=FP_number_num+FP_number
        FN_number_num=FN_number_num+FN_number
        TN_number_num=TN_number_num+TN_number
        
        
        RMSE_PER_NUM=0
        for ii in range(BATCH_SIZE):
            RMSE_PER=(predictions[ii]-actuals[ii]-0.000001)*(predictions[ii]-actuals[ii]-0.000001)
            RMSE_PER_NUM=RMSE_PER_NUM+RMSE_PER
            if ii==(BATCH_SIZE-1):
                RMSE=math.sqrt(RMSE_PER_NUM/BATCH_SIZE)
        '''可以得到每一次100个测试数据的RMSE值'''
        '''
        fpr=FP_number/(FP_number+TN_number+0.000001)
        tpr=TP_number/(TP_number+FN_number+0.000001)
        fpr_record.append(fpr)
        tpr_record.append(tpr)
        '''
        
        
        RMSE_NUM_NUM=RMSE_NUM_NUM+RMSE
        RMSE_RECORD.append(RMSE_NUM_NUM)
        
        if j==189:
            
            precision=TP_number_num/(TP_number_num+FP_number_num+0.00001)
            recall=TP_number_num/(TP_number_num+FN_number_num+0.00001)
            Accuracy=(TP_number_num+TN_number_num)/(TP_number_num+TN_number_num+FP_number_num+FN_number_num+0.000001)
            f1=(2*precision*recall)/(precision+recall+0.00001)    
            RMSE_NUM_NUM=RMSE_NUM_NUM/190
            
            TPR=TP_number_num/(TP_number_num+FN_number_num+0.0000001)
            FPR=FP_number_num/(TN_number_num+FP_number_num+0.0000001)
            
            
            print("acc1是多少",Accuracy)
            print("recall是多少",recall)
            print("precision是多少",precision)
            print("f1是多少",f1)
            print("最终的训练集的RMSE是",RMSE_NUM_NUM)
            
            print(actuals_num.shape)
            acu_curve(predictions_num,actuals_num)
        
            x1=predictions_num
            x2=actuals_num
            #a_csv1=pd.DataFrame(x1)
            #a_csv1.to_excel("predictions_num_cnn_lstm_ATTENTION7888.xls")
            #a_csv2=pd.DataFrame(x2)
            #a_csv2.to_excel("actuals_num_cnn_lstm_ATTENTION7888.xls")
            
plt.figure(1)
#这个RMSE的记录数据是错误的没有什么意义
plt.plot(list1,RMSE_RECORD)



'''loss-epoch 画图'''
x1 = num_epoch

y1 = loss_record
y1_test=loss_test_record


plt.figure(2)
plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-',color='r')

plt.plot(x1, y1, '-',label="Train_loss")
plt.plot(x1,y1_test,'-',label="Test_loss")
plt.title('loss vs. epoches')
plt.xlabel('loss vs. epoches')
plt.ylabel('loss')
plt.legend(loc='best')

plt.show()

plt.figure(3)

x2=num_epoch
y2=test_accuracy_record

plt.plot(x2, y2, '-')
plt.title('Train_accuracy')
plt.show()

'''输出数据'''
a_csv1=pd.DataFrame(test_accuracy_record)
a_csv1.to_excel("测试集准去率CNN-LSTM9.xls")

a_csv2=pd.DataFrame(loss_record)
a_csv2.to_excel("训练集lossCNN-LSTM9.xls")

a_csv3=pd.DataFrame(loss_test_record)
a_csv3.to_excel("测试集的lossCNN-LSTM9.xls")











