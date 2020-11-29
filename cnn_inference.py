import numpy as np
import tensorflow as tf
'''搭建我们要构造 的神经网络的前向传播过程'''
'''定义网络的输入，包括输入的BATCH 的大小，和输入矩阵的长宽高和数据的类型'''


DIM=26
                           
BATCH_SIZE=128           
                     
                           
'''在构建网络之前，先将网络的各个参数确定好'''
'''经过时间窗处理过后的数据是24*26'''
INPUT_NODE=106*DIM
OUT_NODE=2        

'''输入到cnn卷积神经网络的图像的大小'''
IMAGE_SIZE_ROW=106
IMAGE_SIZE_COLUMN=DIM
NUM_CHANNELS=1        
NUM_LABLES=2         

'''在这里给出每一层网络的具体的参数'''
'''第一层卷积层的卷积核的具体的参数，卷积核有四个参数。长宽深度和个数，深度由上一层的深度决定，而个数决定了下一层的深度'''
CONV1_DEEP=32
CONV1_SIZE_ROW=3
CONV1_SIZE_COLUMN=26  

'''给出了第二层的具体的卷积核的参数'''
CONV2_DEEP=32
CONV2_SIZE_ROW=4
CONV2_SIZE_COLUMN=1    
'''通过一维卷积核在一定程度上对特征进行提取'''
'''全连接层节点的个数'''
FC_SIZE=16
'''这里是与NMF提取特征相对应'''
HIDDEN_SIZE=16
'''LSTM C的特征数'''

'''以上数据要根据自己要搭建的网络的具体的数值先给出来'''


'''确定好了网络的结构以后，将网络的前向通路搭建起来，将各个网络分别写出来'''

'''构建前向传播网络函数  inference   '''
'''train 参数用来确定是训练网络，还是来测试网络，regularizer是正则项'''

def inference(input_tensor,train,regularizer):
    
    with tf.variable_scope('layer1-conv1'):  
        conv1_weights=tf.get_variable(
                                          "1111wweights11111111111111",[CONV1_SIZE_ROW,CONV1_SIZE_COLUMN,NUM_CHANNELS,CONV1_DEEP],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1)) 
                                                   
        conv1_bias=tf.get_variable(
                                        "1111wbias11111111111111",[CONV1_DEEP],  
                                        initializer=tf.constant_initializer(0.0))
        print(input_tensor)
             
        '''有了网络的参数后，搭建前向传播网络'''
        '''tf.nn.conv2d,卷积层的计算，第一个变量是一个四维变量，第一维是batch，0代表batch中的第一张照片，然后三位就是正常的卷积层的输入'''
       
        conv1=tf.nn.conv2d(          
                           input_tensor,conv1_weights,strides=[1,1,1,1],padding='VALID')
       
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_bias))
    
    '''在第一个卷积层加上Squeeze层'''
    '''    
    with tf.name_scope('layer-Squeeze_cnn1'):
        
        relu1_shape=relu1.get_shape().as_list()
        Squeeze_conv1_weights1=tf.get_variable(
                                          "sq_fc_1",[relu1_shape[3],2],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        Squeeze_conv1_weights2=tf.get_variable(
                                          "sq_fc_2",[2,relu1_shape[3]],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        F_sq=tf.nn.avg_pool(relu1,ksize=[1,relu1_shape[1],relu1_shape[2],1],strides=[1,1,1,1],padding='VALID')
        F_sq=tf.squeeze(F_sq,[1,2])
        
        
        F_ex=tf.nn.relu(tf.matmul(F_sq,Squeeze_conv1_weights1))
        
        s=tf.nn.sigmoid(tf.matmul(F_ex,Squeeze_conv1_weights2))
        
        print(s)
        print(relu1)
        relu1=tf.reshape(relu1,shape=[relu1_shape[0],relu1_shape[1]*relu1_shape[2],relu1_shape[3]])
        relu1_cc=[]
        for i_c in range(BATCH_SIZE):
            diag_c=tf.diag(s[i_c,:])
            relu1_c=tf.matmul(relu1[i_c,:],diag_c)
            relu1_cc.append(relu1_c)
        relu_next=tf.concat(relu1_cc,axis=0)
        relu_next=tf.reshape(relu_next,shape=[relu1_shape[0],relu1_shape[1],relu1_shape[2],relu1_shape[3]])
        
        
        
        print(relu_next)
       
    '''
    
      
        
    
    '''第二层池化层,ksize池化的大小，strides是它的移动规律'''
    with tf.name_scope('layer2-poolling'):
        pool1=tf.nn.max_pool(
                             relu1,ksize=[1,2,1,1],strides=[1,2,1,1],padding='VALID')
        print(relu1)
        if train:
            pool1=tf.nn.dropout(pool1,0.8)   
    
    
    
    
    ''' 继续构建第三层网络'''
    
    with tf.variable_scope('layer3-conv2'):
        
        conv2_weights=tf.get_variable("1111wweights2222222222222",[CONV2_SIZE_ROW,CONV2_SIZE_COLUMN,CONV1_DEEP,CONV2_DEEP],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_bias=tf.get_variable("1111wbias2222222222222",[CONV2_DEEP],
                                       initializer=tf.constant_initializer(0.0))
        print(pool1)
        conv2=tf.nn.conv2d(
                           pool1,conv2_weights,strides=[1,1,1,1],padding='VALID')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_bias))
        print(relu2)
    '''
    with tf.name_scope('layer-Squeeze_cnn2'):
        
        relu2_shape=relu2.get_shape().as_list()
        Squeeze_conv11_weights=tf.get_variable(
                                          "sq_fc_11",[relu2_shape[3],2],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        Squeeze_conv22_weights=tf.get_variable(
                                          "sq_fc_22",[2,relu2_shape[3]],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        F_sq_2=tf.nn.avg_pool(relu2,ksize=[1,relu2_shape[1],relu2_shape[2],1],strides=[1,1,1,1],padding='VALID')
        F_sq_2=tf.squeeze(F_sq_2,[1,2])
        
        
        F_ex_2=tf.nn.relu(tf.matmul(F_sq_2,Squeeze_conv11_weights))
        s_2=tf.nn.sigmoid(tf.matmul(F_ex_2,Squeeze_conv22_weights))
        
        
        relu2=tf.reshape(relu2,shape=[relu2_shape[0],relu2_shape[1]*relu2_shape[2],relu2_shape[3]])
        relu2_cc=[]
        for i_c_2 in range(BATCH_SIZE):
            diag_c_2=tf.diag(s_2[i_c_2,:])
            relu2_c=tf.matmul(relu2[i_c_2,:],diag_c_2)
            relu2_cc.append(relu2_c)
        relu_next_2=tf.concat(relu2_cc,axis=0)
        relu_next_2=tf.reshape(relu_next_2,shape=[relu2_shape[0],relu2_shape[1],relu2_shape[2],relu2_shape[3]])
        
        
        print(relu_next_2)
    
    
    
    
    
    
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''继续构建一个池化层'''
    with tf.name_scope('layer4-poolling'):
        pool2=tf.nn.max_pool(
                            relu2,ksize=[1,2,1,1],strides=[1,2,1,1],padding='VALID')
        if train:
            pool2=tf.nn.dropout(pool2,0.8) 
            print(pool2)
    '''将CNN得到的特征数据，作为LSTM的输入，输入到LSTM 再将LSTM得到的输出作为全连接层的输入，得到最终的分类结果'''
   
    
        
    
    
    '''将一个矩阵拉伸成一个向量，把数据输入到全连接层'''
    '''
    pool_shape=pool2.get_shape().as_list()
    print(pool_shape)
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    print(nodes) 
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])
    lstm_output=reshaped;
    
    '''
    
    
    '''栈式双向LSTM'''
    
    '''
    pool_shape=pool2.get_shape().as_list()
    nodes=pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(pool2,[pool_shape[0],pool_shape[1],nodes])
    
    
    
    
    TIMESTEPS=pool_shape[1]
    with tf.variable_scope('Bi-LSTM'):
        seq_lengths=[TIMESTEPS]*BATCH_SIZE
        stacked_rnn = []
        stacked_bw_rnn = []
        
        for i in range(3):
            stacked_rnn.append(tf.contrib.rnn.LSTMCell(HIDDEN_SIZE))
            stacked_bw_rnn.append(tf.contrib.rnn.LSTMCell(HIDDEN_SIZE))

        mcell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
        mcell_bw = tf.contrib.rnn.MultiRNNCell(stacked_bw_rnn)    

        bioutputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([mcell],[mcell_bw],reshaped,sequence_length=seq_lengths, dtype=tf.float32)
        # bioutputs [BATCH_SIZE,STEPS,HIDDENSIZE(前后连接以后的)] (第一步到最后一步的h)                                                                                       
        
        #这仅仅是最后一层的h 而不是lstm 的输出y
        lstm_output=bioutputs[:,-1,:]
    
    nodes=2*HIDDEN_SIZE
    '''
    
    '''加入attention机制'''
    '''
    cnn_num=12
    #state_T=tf.transpose(state, perm=[2, 3, 0,1])#batch_size*hidensize*step
    #state_T=state_T[:,:,:,-1]#128*16*1
    
    with tf.variable_scope('lstm-Attention'):
        seq_size=bioutputs.shape[1].value#记录有多少个时间步
        seq_hidden_size=bioutputs.shape[2].value#相当于hiddensize

        
        atten_conv1_weights=tf.get_variable(
                                          "atten-cnn1",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv2_weights=tf.get_variable(
                                          "atten-cnn2",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv3_weights=tf.get_variable(
                                          "atten-cnn3",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv4_weights=tf.get_variable(
                                          "atten-cnn4",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        atten_conv5_weights=tf.get_variable(
                                          "atten-cnn5",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv6_weights=tf.get_variable(
                                          "atten-cnn6",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv7_weights=tf.get_variable(
                                          "atten-cnn7",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv8_weights=tf.get_variable(
                                          "atten-cnn8",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        
        bioutputs_Keep=bioutputs
        bioutputs_T=tf.transpose(bioutputs, perm=[0,2,1])#128*hidden_size*steps
        bioutputs_T_Keep=bioutputs
        bioutputs_T=bioutputs_T[:,:,0:-1]
        bioutputs_T=tf.expand_dims(bioutputs_T, 3)#128*hidden_size*steps-1*1
        
        cnn_HC=[]
        
        cnn_1=tf.nn.conv2d(
                           bioutputs_T,atten_conv1_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_1)
        cnn_2=tf.nn.conv2d(
                           bioutputs_T,atten_conv2_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_2)
        cnn_3=tf.nn.conv2d(
                           bioutputs_T,atten_conv3_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_3)
        cnn_4=tf.nn.conv2d(
                           bioutputs_T,atten_conv4_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_4)
        cnn_5=tf.nn.conv2d(
                           bioutputs_T,atten_conv5_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_5)
        cnn_6=tf.nn.conv2d(
                           bioutputs_T,atten_conv6_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_6)
        cnn_7=tf.nn.conv2d(
                           bioutputs_T,atten_conv7_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_7)
        cnn_8=tf.nn.conv2d(
                           bioutputs_T,atten_conv8_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_8)
        
        
        
        
        
        
        
        cnn_HC=tf.concat(cnn_HC,axis=2)#128*hidden_size*cnn_num*1
        print(cnn_HC)
        
        w1=tf.get_variable("attention1",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w2=tf.get_variable("attention2",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w3=tf.get_variable("attention3",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w4=tf.get_variable("attention4",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w5=tf.get_variable("attention5",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w6=tf.get_variable("attention6",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w7=tf.get_variable("attention7",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w8=tf.get_variable("attention8",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w9=tf.get_variable("attention9",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w10=tf.get_variable("attention10",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w11=tf.get_variable("attention11",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w12=tf.get_variable("attention12",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w13=tf.get_variable("attention13",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w14=tf.get_variable("attention14",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w15=tf.get_variable("attention15",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w16=tf.get_variable("attention16",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w17=tf.get_variable("attention17",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w18=tf.get_variable("attention18",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w19=tf.get_variable("attention19",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w20=tf.get_variable("attention20",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w21=tf.get_variable("attention21",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w22=tf.get_variable("attention22",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w23=tf.get_variable("attention23",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w24=tf.get_variable("attention24",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w25=tf.get_variable("attention25",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w26=tf.get_variable("attention26",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w27=tf.get_variable("attention27",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w28=tf.get_variable("attention28",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w29=tf.get_variable("attention29",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w30=tf.get_variable("attention30",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w31=tf.get_variable("attention31",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w32=tf.get_variable("attention32",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        
        
        
        
        
        
        
    
        
        
        VT=[]
        for iii in range(BATCH_SIZE):
            
            a=[]
            bioutputs[iii,-1,:]
            
            S_T=tf.expand_dims(bioutputs_Keep[iii,-1,:],1)#hidden_szie*1
            
            #cnn_HC=tf.squeeze(cnn_HC, [3])
            cnn_HC_1=tf.expand_dims(cnn_HC[iii,0], 0)#1*cnn_num*1
            cnn_HC_1=tf.squeeze(cnn_HC_1, [2])#1*cnn_num
            a1_chushi=tf.matmul(cnn_HC_1,w1)#1*hidden_size
            a1=tf.sigmoid(tf.matmul(a1_chushi,S_T))#1*1
            a.append(a1) 
            cnn_HC_2=tf.expand_dims(cnn_HC[iii,1], 0)
            cnn_HC_2=tf.squeeze(cnn_HC_2, [2])
            a2_chushi=tf.matmul(cnn_HC_2,w2)
            a2=tf.sigmoid(tf.matmul(a2_chushi,S_T))
            a.append(a2) 
            cnn_HC_3=tf.expand_dims(cnn_HC[iii,2], 0)
            cnn_HC_3=tf.squeeze(cnn_HC_3, [2])
            a3_chushi=tf.matmul(cnn_HC_3,w3)
            a3=tf.sigmoid(tf.matmul(a3_chushi,S_T))
            a.append(a3) 
            cnn_HC_4=tf.expand_dims(cnn_HC[iii,3], 0)
            cnn_HC_4=tf.squeeze(cnn_HC_4, [2])
            a4_chushi=tf.matmul(cnn_HC_4,w4)
            a4=tf.sigmoid(tf.matmul(a4_chushi,S_T))
            a.append(a4) 
            cnn_HC_5=tf.expand_dims(cnn_HC[iii,4], 0)
            cnn_HC_5=tf.squeeze(cnn_HC_5, [2])
            a5_chushi=tf.matmul(cnn_HC_5,w5)
            a5=tf.sigmoid(tf.matmul(a5_chushi,S_T))
            a.append(a5) 
            
            cnn_HC_6=tf.expand_dims(cnn_HC[iii,5], 0)
            cnn_HC_6=tf.squeeze(cnn_HC_6, [2])
            a6_chushi=tf.matmul(cnn_HC_6,w6)
            a6=tf.sigmoid(tf.matmul(a6_chushi,S_T))
            a.append(a6) 
            
            cnn_HC_7=tf.expand_dims(cnn_HC[iii,6], 0)
            cnn_HC_7=tf.squeeze(cnn_HC_7, [2])
            a7_chushi=tf.matmul(cnn_HC_7,w7)
            a7=tf.sigmoid(tf.matmul(a7_chushi,S_T))
            a.append(a7) 
            cnn_HC_8=tf.expand_dims(cnn_HC[iii,7], 0)
            cnn_HC_8=tf.squeeze(cnn_HC_8, [2])
            a8_chushi=tf.matmul(cnn_HC_8,w8)
            a8=tf.sigmoid(tf.matmul(a8_chushi,S_T))
            a.append(a8) 
            cnn_HC_9=tf.expand_dims(cnn_HC[iii,8], 0)
            cnn_HC_9=tf.squeeze(cnn_HC_9, [2])
            a9_chushi=tf.matmul(cnn_HC_9,w9)
            a9=tf.sigmoid(tf.matmul(a9_chushi,S_T))
            a.append(a9) 
            cnn_HC_10=tf.expand_dims(cnn_HC[iii,9], 0)
            cnn_HC_10=tf.squeeze(cnn_HC_10, [2])
            a10_chushi=tf.matmul(cnn_HC_10,w10)
            a10=tf.sigmoid(tf.matmul(a10_chushi,S_T))
            a.append(a10) 
            cnn_HC_11=tf.expand_dims(cnn_HC[iii,10], 0)
            cnn_HC_11=tf.squeeze(cnn_HC_11, [2])
            a11_chushi=tf.matmul(cnn_HC_11,w11)
            a11=tf.sigmoid(tf.matmul(a11_chushi,S_T))
            a.append(a11) 
            cnn_HC_12=tf.expand_dims(cnn_HC[iii,11], 0)
            cnn_HC_12=tf.squeeze(cnn_HC_12, [2])
            a12_chushi=tf.matmul(cnn_HC_12,w12)
            a12=tf.sigmoid(tf.matmul(a12_chushi,S_T))
            a.append(a12) 
            cnn_HC_13=tf.expand_dims(cnn_HC[iii,12], 0)
            cnn_HC_13=tf.squeeze(cnn_HC_13, [2])
            a13_chushi=tf.matmul(cnn_HC_13,w13)
            a13=tf.sigmoid(tf.matmul(a13_chushi,S_T))
            a.append(a13) 
            cnn_HC_14=tf.expand_dims(cnn_HC[iii,13], 0)
            cnn_HC_14=tf.squeeze(cnn_HC_14, [2])
            a14_chushi=tf.matmul(cnn_HC_14,w14)
            a14=tf.sigmoid(tf.matmul(a14_chushi,S_T))
            a.append(a14) 
            cnn_HC_15=tf.expand_dims(cnn_HC[iii,14], 0)
            cnn_HC_15=tf.squeeze(cnn_HC_15, [2])
            a15_chushi=tf.matmul(cnn_HC_15,w15)
            a15=tf.sigmoid(tf.matmul(a15_chushi,S_T))
            a.append(a15) 
            cnn_HC_16=tf.expand_dims(cnn_HC[iii,15], 0)
            cnn_HC_16=tf.squeeze(cnn_HC_16, [2])
            a16_chushi=tf.matmul(cnn_HC_16,w16)
            a16=tf.sigmoid(tf.matmul(a16_chushi,S_T))
            a.append(a16) 
            
            cnn_HC_17=tf.expand_dims(cnn_HC[iii,16], 0)#1*cnn_num*1
            cnn_HC_17=tf.squeeze(cnn_HC_17, [2])#1*cnn_num
            a17_chushi=tf.matmul(cnn_HC_17,w17)#1*hidden_size
            a17=tf.sigmoid(tf.matmul(a17_chushi,S_T))#1*1
            a.append(a17)
            
            cnn_HC_18=tf.expand_dims(cnn_HC[iii,17], 0)#1*cnn_num*1
            cnn_HC_18=tf.squeeze(cnn_HC_18, [2])#1*cnn_num
            a18_chushi=tf.matmul(cnn_HC_18,w18)#1*hidden_size
            a18=tf.sigmoid(tf.matmul(a18_chushi,S_T))#1*1
            a.append(a18)
            
            cnn_HC_19=tf.expand_dims(cnn_HC[iii,18], 0)#1*cnn_num*1
            cnn_HC_19=tf.squeeze(cnn_HC_19, [2])#1*cnn_num
            a19_chushi=tf.matmul(cnn_HC_19,w19)#1*hidden_size
            a19=tf.sigmoid(tf.matmul(a19_chushi,S_T))#1*1
            a.append(a19)
            
            cnn_HC_20=tf.expand_dims(cnn_HC[iii,19], 0)#1*cnn_num*1
            cnn_HC_20=tf.squeeze(cnn_HC_20, [2])#1*cnn_num
            a20_chushi=tf.matmul(cnn_HC_20,w20)#1*hidden_size
            a20=tf.sigmoid(tf.matmul(a20_chushi,S_T))#1*1
            a.append(a20)
            
            
            cnn_HC_21=tf.expand_dims(cnn_HC[iii,20], 0)#1*cnn_num*1
            cnn_HC_21=tf.squeeze(cnn_HC_21, [2])#1*cnn_num
            a21_chushi=tf.matmul(cnn_HC_21,w21)#1*hidden_size
            a21=tf.sigmoid(tf.matmul(a21_chushi,S_T))#1*1
            a.append(a21)
            
            cnn_HC_22=tf.expand_dims(cnn_HC[iii,21], 0)#1*cnn_num*1
            cnn_HC_22=tf.squeeze(cnn_HC_22, [2])#1*cnn_num
            a22_chushi=tf.matmul(cnn_HC_22,w22)#1*hidden_size
            a22=tf.sigmoid(tf.matmul(a22_chushi,S_T))#1*1
            a.append(a22)
            
            cnn_HC_23=tf.expand_dims(cnn_HC[iii,22], 0)#1*cnn_num*1
            cnn_HC_23=tf.squeeze(cnn_HC_23, [2])#1*cnn_num
            a23_chushi=tf.matmul(cnn_HC_23,w23)#1*hidden_size
            a23=tf.sigmoid(tf.matmul(a23_chushi,S_T))#1*1
            a.append(a23)
            
            cnn_HC_24=tf.expand_dims(cnn_HC[iii,23], 0)#1*cnn_num*1
            cnn_HC_24=tf.squeeze(cnn_HC_24, [2])#1*cnn_num
            a24_chushi=tf.matmul(cnn_HC_24,w24)#1*hidden_size
            a24=tf.sigmoid(tf.matmul(a24_chushi,S_T))#1*1
            a.append(a24)
            
            cnn_HC_25=tf.expand_dims(cnn_HC[iii,24], 0)#1*cnn_num*1
            cnn_HC_25=tf.squeeze(cnn_HC_25, [2])#1*cnn_num
            a25_chushi=tf.matmul(cnn_HC_25,w25)#1*hidden_size
            a25=tf.sigmoid(tf.matmul(a25_chushi,S_T))#1*1
            a.append(a25)
            
            cnn_HC_26=tf.expand_dims(cnn_HC[iii,25], 0)#1*cnn_num*1
            cnn_HC_26=tf.squeeze(cnn_HC_26, [2])#1*cnn_num
            a26_chushi=tf.matmul(cnn_HC_26,w26)#1*hidden_size
            a26=tf.sigmoid(tf.matmul(a26_chushi,S_T))#1*1
            a.append(a26)
            
            cnn_HC_27=tf.expand_dims(cnn_HC[iii,26], 0)#1*cnn_num*1
            cnn_HC_27=tf.squeeze(cnn_HC_27, [2])#1*cnn_num
            a27_chushi=tf.matmul(cnn_HC_27,w27)#1*hidden_size
            a27=tf.sigmoid(tf.matmul(a27_chushi,S_T))#1*1
            a.append(a27)
            
            cnn_HC_28=tf.expand_dims(cnn_HC[iii,27], 0)#1*cnn_num*1
            cnn_HC_28=tf.squeeze(cnn_HC_28, [2])#1*cnn_num
            a28_chushi=tf.matmul(cnn_HC_28,w28)#1*hidden_size
            a28=tf.sigmoid(tf.matmul(a28_chushi,S_T))#1*1
            a.append(a28)
            
            cnn_HC_29=tf.expand_dims(cnn_HC[iii,28], 0)#1*cnn_num*1
            cnn_HC_29=tf.squeeze(cnn_HC_29, [2])#1*cnn_num
            a29_chushi=tf.matmul(cnn_HC_29,w29)#1*hidden_size
            a29=tf.sigmoid(tf.matmul(a29_chushi,S_T))#1*1
            a.append(a29)
            
            cnn_HC_30=tf.expand_dims(cnn_HC[iii,29], 0)#1*cnn_num*1
            cnn_HC_30=tf.squeeze(cnn_HC_30, [2])#1*cnn_num
            a30_chushi=tf.matmul(cnn_HC_30,w30)#1*hidden_size
            a30=tf.sigmoid(tf.matmul(a30_chushi,S_T))#1*1
            a.append(a30)
            
            cnn_HC_31=tf.expand_dims(cnn_HC[iii,30], 0)#1*cnn_num*1
            cnn_HC_31=tf.squeeze(cnn_HC_31, [2])#1*cnn_num
            a31_chushi=tf.matmul(cnn_HC_31,w31)#1*hidden_size
            a31=tf.sigmoid(tf.matmul(a31_chushi,S_T))#1*1
            a.append(a31)
            
            cnn_HC_32=tf.expand_dims(cnn_HC[iii,31], 0)#1*cnn_num*1
            cnn_HC_32=tf.squeeze(cnn_HC_32, [2])#1*cnn_num
            a32_chushi=tf.matmul(cnn_HC_32,w32)#1*hidden_size
            a32=tf.sigmoid(tf.matmul(a32_chushi,S_T))#1*1
            a.append(a32)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            a=tf.concat(a,axis=0)
            a=tf.transpose(a, perm=[1,0])#1*16 
            
            
            
            cnn_HC_V=tf.squeeze(cnn_HC[iii,:,:], [2])#hidden_size*cnn_num
            
            
            vt=tf.matmul(a,cnn_HC_V)#1*cnn_num
            VT=tf.concat(vt,axis=0)#128*cnn_num
        
        
        
        wv=tf.get_variable("attention_wv",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        
        wh=tf.get_variable("attention_wh",[seq_hidden_size,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        htt=tf.matmul(VT,wv)+tf.matmul(bioutputs_Keep[:,-1,:],wh)#128*hidden_size 的数据
        whh=tf.get_variable("attention_whh",[seq_hidden_size,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        lstm_output_attention=tf.matmul(htt,whh)
    
        nodes=seq_hidden_size
        
    
    
    
    
    
    
    
    
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pool_shape=pool2.get_shape().as_list()
    
    nodes=pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(pool2,[pool_shape[0],pool_shape[1],nodes])
    TIMESTEPS=pool_shape[1]
    
    
    NUM_LAYERS=1
    with tf.variable_scope('CNN-LSTM'):
    
        cell=tf.nn.rnn_cell.MultiRNNCell(
                                    [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)for _ in range(NUM_LAYERS)])
        outputs,state=tf.nn.dynamic_rnn(cell,reshaped,dtype=tf.float32)
        
        lstm_output=outputs[:,-1,:]#128*16
        
    nodes=HIDDEN_SIZE

    
    
    '''这里 实现有关注意力机制的网络'''
    '''卷积核的个数是超参数'''
    '''
    cnn_num=4
    state_T=tf.transpose(state, perm=[2, 3, 0,1])#batch_size*hidensize*step
    state_T=state_T[:,:,:,-1]#128*16*1
    state_T=tf.reshape(state_T,[BATCH_SIZE,16,1])
    with tf.variable_scope('lstm-Attention'):
        seq_size=outputs.shape[1].value#记录有多少个时间步
        seq_hidden_size=outputs.shape[2].value#相当于hiddensize

        
        atten_conv1_weights=tf.get_variable(
                                          "atten-cnn1",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv2_weights=tf.get_variable(
                                          "atten-cnn2",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv3_weights=tf.get_variable(
                                          "atten-cnn3",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv4_weights=tf.get_variable(
                                          "atten-cnn4",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        atten_conv5_weights=tf.get_variable(
                                          "atten-cnn5",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        atten_conv6_weights=tf.get_variable(
                                          "atten-cnn6",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv7_weights=tf.get_variable(
                                          "atten-cnn7",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv8_weights=tf.get_variable(
                                          "atten-cnn8",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        atten_conv9_weights=tf.get_variable(
                                          "atten-cnn9",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        atten_conv10_weights=tf.get_variable(
                                          "atten-cnn10",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv11_weights=tf.get_variable(
                                          "atten-cnn11",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv12_weights=tf.get_variable(
                                          "atten-cnn12",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        atten_conv13_weights=tf.get_variable(
                                          "atten-cnn13",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        atten_conv14_weights=tf.get_variable(
                                          "atten-cnn14",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv15_weights=tf.get_variable(
                                          "atten-cnn15",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        atten_conv16_weights=tf.get_variable(
                                          "atten-cnn16",[1,seq_size-1,1,1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        
        outputs_T=tf.transpose(outputs, perm=[0,2,1])#128*hidden_size*steps
        outputs_T=outputs_T[:,:,0:-1]
        outputs_T=tf.expand_dims(outputs_T, 3)#128*hidden_size*steps-1*1
        
        cnn_HC=[]
        
        cnn_1=tf.nn.conv2d(
                           outputs_T,atten_conv1_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_1)
        cnn_2=tf.nn.conv2d(
                           outputs_T,atten_conv2_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_2)
        cnn_3=tf.nn.conv2d(
                           outputs_T,atten_conv3_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_3)
        cnn_4=tf.nn.conv2d(
                           outputs_T,atten_conv4_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_4)
        
        cnn_5=tf.nn.conv2d(
                           outputs_T,atten_conv5_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_5)
        cnn_6=tf.nn.conv2d(
                           outputs_T,atten_conv6_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_6)
        cnn_7=tf.nn.conv2d(
                           outputs_T,atten_conv7_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_7)
        cnn_8=tf.nn.conv2d(
                           outputs_T,atten_conv8_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_8)
        
        cnn_9=tf.nn.conv2d(
                           outputs_T,atten_conv9_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_9)
        cnn_10=tf.nn.conv2d(
                           outputs_T,atten_conv10_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_10)
        cnn_11=tf.nn.conv2d(
                           outputs_T,atten_conv11_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_11)
        cnn_12=tf.nn.conv2d(
                           outputs_T,atten_conv12_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_12)
        
        
        cnn_13=tf.nn.conv2d(
                           outputs_T,atten_conv13_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_13)
        cnn_14=tf.nn.conv2d(
                           outputs_T,atten_conv14_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_14)
        cnn_15=tf.nn.conv2d(
                           outputs_T,atten_conv15_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_15)
        cnn_16=tf.nn.conv2d(
                           outputs_T,atten_conv16_weights,strides=[1,1,1,1],padding='VALID')
        cnn_HC.append(cnn_16)
        
        
        
        
        
        
        
        cnn_HC=tf.concat(cnn_HC,axis=2)#128*hidden_size*cnn_num*1
        print(cnn_HC)
        
        w1=tf.get_variable("attention1",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        w2=tf.get_variable("attention2",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w3=tf.get_variable("attention3",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w4=tf.get_variable("attention4",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w5=tf.get_variable("attention5",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w6=tf.get_variable("attention6",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w7=tf.get_variable("attention7",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w8=tf.get_variable("attention8",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w9=tf.get_variable("attention9",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w10=tf.get_variable("attention10",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w11=tf.get_variable("attention11",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w12=tf.get_variable("attention12",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w13=tf.get_variable("attention13",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w14=tf.get_variable("attention14",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w15=tf.get_variable("attention15",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        w16=tf.get_variable("attention16",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        
        VT=[]
        for iii in range(BATCH_SIZE):
            
            a=[]
            S_T=tf.expand_dims(state_T[iii,:,-1],1)#hidden_szie*1
            #cnn_HC=tf.squeeze(cnn_HC, [3])
            cnn_HC_1=tf.expand_dims(cnn_HC[iii,0], 0)#1*cnn_num*1
            cnn_HC_1=tf.squeeze(cnn_HC_1, [2])#1*cnn_num
            a1_chushi=tf.matmul(cnn_HC_1,w1)#1*hidden_size
            a1=tf.sigmoid(tf.matmul(a1_chushi,S_T))#1*1
            a.append(a1) 
            cnn_HC_2=tf.expand_dims(cnn_HC[iii,1], 0)
            cnn_HC_2=tf.squeeze(cnn_HC_2, [2])
            a2_chushi=tf.matmul(cnn_HC_2,w1)
            a2=tf.sigmoid(tf.matmul(a2_chushi,S_T))
            a.append(a2) 
            cnn_HC_3=tf.expand_dims(cnn_HC[iii,2], 0)
            cnn_HC_3=tf.squeeze(cnn_HC_3, [2])
            a3_chushi=tf.matmul(cnn_HC_3,w1)
            a3=tf.sigmoid(tf.matmul(a3_chushi,S_T))
            a.append(a3) 
            cnn_HC_4=tf.expand_dims(cnn_HC[iii,3], 0)
            cnn_HC_4=tf.squeeze(cnn_HC_4, [2])
            a4_chushi=tf.matmul(cnn_HC_4,w1)
            a4=tf.sigmoid(tf.matmul(a4_chushi,S_T))
            a.append(a4) 
            cnn_HC_5=tf.expand_dims(cnn_HC[iii,4], 0)
            cnn_HC_5=tf.squeeze(cnn_HC_5, [2])
            a5_chushi=tf.matmul(cnn_HC_5,w1)
            a5=tf.sigmoid(tf.matmul(a5_chushi,S_T))
            a.append(a5) 
            
            cnn_HC_6=tf.expand_dims(cnn_HC[iii,5], 0)
            cnn_HC_6=tf.squeeze(cnn_HC_6, [2])
            a6_chushi=tf.matmul(cnn_HC_6,w1)
            a6=tf.sigmoid(tf.matmul(a6_chushi,S_T))
            a.append(a6) 
            
            cnn_HC_7=tf.expand_dims(cnn_HC[iii,6], 0)
            cnn_HC_7=tf.squeeze(cnn_HC_7, [2])
            a7_chushi=tf.matmul(cnn_HC_7,w1)
            a7=tf.sigmoid(tf.matmul(a7_chushi,S_T))
            a.append(a7) 
            cnn_HC_8=tf.expand_dims(cnn_HC[iii,7], 0)
            cnn_HC_8=tf.squeeze(cnn_HC_8, [2])
            a8_chushi=tf.matmul(cnn_HC_8,w1)
            a8=tf.sigmoid(tf.matmul(a8_chushi,S_T))
            a.append(a8) 
            cnn_HC_9=tf.expand_dims(cnn_HC[iii,8], 0)
            cnn_HC_9=tf.squeeze(cnn_HC_9, [2])
            a9_chushi=tf.matmul(cnn_HC_9,w1)
            a9=tf.sigmoid(tf.matmul(a9_chushi,S_T))
            a.append(a9) 
            cnn_HC_10=tf.expand_dims(cnn_HC[iii,9], 0)
            cnn_HC_10=tf.squeeze(cnn_HC_10, [2])
            a10_chushi=tf.matmul(cnn_HC_10,w1)
            a10=tf.sigmoid(tf.matmul(a10_chushi,S_T))
            a.append(a10) 
            cnn_HC_11=tf.expand_dims(cnn_HC[iii,10], 0)
            cnn_HC_11=tf.squeeze(cnn_HC_11, [2])
            a11_chushi=tf.matmul(cnn_HC_11,w1)
            a11=tf.sigmoid(tf.matmul(a11_chushi,S_T))
            a.append(a11) 
            cnn_HC_12=tf.expand_dims(cnn_HC[iii,11], 0)
            cnn_HC_12=tf.squeeze(cnn_HC_12, [2])
            a12_chushi=tf.matmul(cnn_HC_12,w1)
            a12=tf.sigmoid(tf.matmul(a12_chushi,S_T))
            a.append(a12) 
            cnn_HC_13=tf.expand_dims(cnn_HC[iii,12], 0)
            cnn_HC_13=tf.squeeze(cnn_HC_13, [2])
            a13_chushi=tf.matmul(cnn_HC_13,w1)
            a13=tf.sigmoid(tf.matmul(a13_chushi,S_T))
            a.append(a13) 
            cnn_HC_14=tf.expand_dims(cnn_HC[iii,13], 0)
            cnn_HC_14=tf.squeeze(cnn_HC_14, [2])
            a14_chushi=tf.matmul(cnn_HC_14,w1)
            a14=tf.sigmoid(tf.matmul(a14_chushi,S_T))
            a.append(a14) 
            cnn_HC_15=tf.expand_dims(cnn_HC[iii,14], 0)
            cnn_HC_15=tf.squeeze(cnn_HC_15, [2])
            a15_chushi=tf.matmul(cnn_HC_15,w1)
            a15=tf.sigmoid(tf.matmul(a15_chushi,S_T))
            a.append(a15) 
            cnn_HC_16=tf.expand_dims(cnn_HC[iii,15], 0)
            cnn_HC_16=tf.squeeze(cnn_HC_16, [2])
            a16_chushi=tf.matmul(cnn_HC_16,w1)
            a16=tf.sigmoid(tf.matmul(a16_chushi,S_T))
            a.append(a16) 
            a=tf.concat(a,axis=0)
            a=tf.transpose(a, perm=[1,0])#1*16 
            
            
            
            cnn_HC_V=tf.squeeze(cnn_HC[iii,:,:], [2])#hidden_size*cnn_num
            
            
            vt=tf.matmul(a,cnn_HC_V)#1*cnn_num
            VT=tf.concat(vt,axis=0)#128*cnn_num
        
        
        
        wv=tf.get_variable("attention_wv",[cnn_num,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        
        wh=tf.get_variable("attention_wh",[seq_hidden_size,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        htt=tf.matmul(VT,wv)+tf.matmul(state_T[:,:,-1],wh)#128*hidden_size 的数据
        whh=tf.get_variable("attention_whh",[seq_hidden_size,seq_hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        lstm_output_attention=tf.matmul(htt,whh)
    
        nodes=seq_hidden_size
        
    '''
    
    
    
    
    
     
    '''dropout层，对某一层的网络使用了dropout 处理了以后，这一层的参数，会有选择的在每一次进行训练，从而能够学到数据更深层次的特征，减少过拟合现象的发生'''
    '''这里在全连接层做一个示范，在训练全连接层的时候，采用dropout技术'''
    
    with tf.variable_scope('layer5-fcl'):
        
        fc1_weights=tf.get_variable("1111wweight333333333333",[nodes,FC_SIZE],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
                                                 
        fc1_biaes=tf.get_variable(
                                       "1111wbias333333333333",[FC_SIZE],
                                       initializer=tf.constant_initializer(0.1))
            
        '''对全连接层的参数加入正则化要求'''
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1=tf.nn.sigmoid(tf.matmul(lstm_output,fc1_weights)+fc1_biaes)
        #fc1=tf.matmul(lstm_output,fc1_weights)+fc1_biaes
        '''fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biaes)'''
        '''使用dropout层'''
        
        if train:
            fc1=tf.nn.dropout(fc1,0.8)
           
        '''最后一层全连接层，这一层的数据经过softmax后得到分类的结果'''
    with tf.variable_scope('layer6_fcl'):
       
        fc2_weights=tf.get_variable(
                                         "1111wweight1444444444444",[FC_SIZE,NUM_LABLES],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        fc2_biaes=tf.get_variable(
                                      "1111wbias1444444444444",[NUM_LABLES],
                                      initializer=tf.constant_initializer(0.1))
        
            
        if regularizer !=None:
             tf.add_to_collection('losses',regularizer(fc2_weights))
        logit=tf.matmul(fc1,fc2_weights)+fc2_biaes
        #logit_sigmoid=tf.sigmoid(logit)
        print("输出序列的形状是",logit.shape)

    
    
    return logit














