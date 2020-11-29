import numpy as np
import pandas as pd
from sklearn import cross_validation

import xlwt
import random
DIM=26
from sklearn import preprocessing
'''数据预处理部分将数据通过时间窗得到CNN所需要的数据'''
'''w表示时间的跨度，s表示每次时间窗的横移距离'''
def time_window(x,y,w,s,shurushujugeshu): 
    shuliang=int(((shurushujugeshu-w)/12)-1)
    '''shuliang=int(((shurushujugeshu-w)/s)-1)'''
    features=np.zeros((shuliang,w,DIM))
    lables=np.zeros((shuliang,1))
    for i in range(0,shuliang):
        for j in range(0,w):
            features[i,j,:]=x[i*s+j,:]
            if j==(w-1):
                lables[i,:]=y[i*s+j,:]
    '''使用distance image 来构成图像'''
    '''
    print('22222222222222222222',features.shape)
    features_m=np.zeros((shuliang,DIM,DIM))
    for w1 in range(0,shuliang):
        for i1 in range(0,DIM):
            for j1 in range(0,DIM):
                num_help=0
                for qq in range(0,48):
                    num_help=num_help+abs(features[w1,qq,i1]-features[w1,qq,j1])
                features_m[w1,i1,j1]=num_help/48
    
    print('yyyyyyyyyyyyyyyyyyyyy',features_m.shape)
    
    '''
    
    
    return features,lables














def data_perprocessing():
    
    x1=pd.read_excel('最大值MAX.xls')
    height,width = x1.shape
    x_max = np.zeros([height,width-1])
    for i in range(height):
        for j in range(width-1):
            x_max[i,j]  = x1.iloc[i,j+1]
    
    x2=pd.read_excel('最小值MIN.xls')
    height,width = x2.shape
    x_min = np.zeros([height,width-1])
    for i in range(height):
        for j in range(width-1):
            x_min[i,j]  = x2.iloc[i,j+1]
            
    x3=pd.read_excel('差距DISTANCE.xls')
    height,width = x3.shape
    x_distance = np.zeros([height,width-1])
    for i in range(height):
        for j in range(width-1):
            x_distance[i,j]  = x3.iloc[i,j+1]
    
    column_min=x_min.T
    column_distance=x_distance.T




    x2=pd.read_csv('label_y_failure.csv')
    height,width = x2.shape
    label_y_failure = np.zeros([height,width])
    for i in range(height):
        for j in range(width):
            label_y_failure[i,j]  = x2.iloc[i,j]
    print(label_y_failure.shape)

    x2=pd.read_csv('label_y_normal.csv')
    height,width = x2.shape
    label_y_normal= np.zeros([height,width])
    for i in range(height):
        for j in range(width):
            label_y_normal[i,j]  = x2.iloc[i,j]
    print(label_y_normal.shape)

    x2=pd.read_csv('test_x.csv')
    height,width = x2.shape
    test_x= np.zeros([height,width])
    for i in range(height):
        for j in range(width):
            test_x[i,j]  = x2.iloc[i,j]
    print(test_x.shape)
    test_x=test_x.reshape([25809,106,26])
    print(test_x.shape)
    x2=pd.read_csv('test_y.csv')
    height,width = x2.shape
    test_y= np.zeros([height,width])
    for i in range(height):
        for j in range(width):
            test_y[i,j]  = x2.iloc[i,j]
    print(test_y.shape)

    x2=pd.read_csv('train_x_failure.csv')
    height,width = x2.shape
    train_x_failure= np.zeros([height,width])
    for i in range(height):
        for j in range(width):
            train_x_failure[i,j]  = x2.iloc[i,j]
    print(train_x_failure.shape)
    train_x_failure=train_x_failure.reshape([height,106,26])
    x2=pd.read_csv('train_x_normal.csv')
    height,width = x2.shape
    train_x_normal= np.zeros([height,width-1])
    for i in range(height):
        for j in range(width-1):
            train_x_normal[i,j]  = x2.iloc[i,j+1]
    print(train_x_normal.shape)
    train_x_normal=train_x_normal.reshape([height,106,26])
    print(train_x_normal.shape)

    train_x=np.concatenate([train_x_normal,train_x_failure])
    train_y=np.concatenate([label_y_normal,label_y_failure])

    print(train_x.shape)
    print(train_y.shape)
#对数据进行归一化处理

    a,b,c=train_x.shape
    a1,b1,c1=test_x.shape
    for i in range(a):
        for j in range(b):
            for k in range(c):
                train_x[i,j,k]=(train_x[i,j,k]-column_min[0,k])/column_distance[0,k]
    for i in range(a1):
        for j in range(b1):
            for k in range(c1):
                test_x[i,j,k]=(test_x[i,j,k]-column_min[0,k])/column_distance[0,k]


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    x2=pd.read_csv('label_y_failure.csv')
    height,width = x2.shape
    x_min = np.zeros([height,width-1])
    for i in range(height):
        for j in range(width-1):
            x_min[i,j]  = x2.iloc[i,j+1]
    
    
    
    
    
    df_feature1=pd.read_excel('feature.xlsx')
    height1,width1 = df_feature1.shape
    x1 = np.zeros([height1,width1])
    print(height1,width1,type(df_feature1))
    for i in range(height1):
        for j in range(width1): 
            x1[i,j]  = df_feature1.iloc[i,j]
    print(x1.shape)
    #x1=preprocessing.scale(x1)
    
    column_max=x1.max(axis=0)
    column_min=x1.min(axis=0)
    column_distance=column_max-column_min
    b_csv1=pd.DataFrame(column_max)
    b_csv1.to_excel("MAX.xls")
    b_csv2=pd.DataFrame(column_min)
    b_csv2.to_excel("MIN.xls")
    b_csv3=pd.DataFrame(column_distance)
    b_csv3.to_excel("DISTANCE.xls")
    
    x_15=np.zeros([374100,26])
    for i in range(374100):
        x_15[i,:]=x1[i,:]
    x_21=np.zeros([179567,26])
    for j in range(179567):
        x_21[j,:]=x1[j+374101,:]    
    
    
    
    
    df_feature=pd.read_excel('151515features.xlsx')
    height,width = df_feature.shape
    x_15 = np.zeros([height,width])
    print(height,width,type(df_feature))
    for i in range(height):
        for j in range(width): 
            x_15[i,j]  = df_feature.iloc[i,j]
    print(x_15.shape)
    
    
    column_max=x_15.max(axis=0)
    column_min=x_15.min(axis=0)
    column_distance=column_max-column_min
    
    
    
    for i in range(0,height):
        for j in range(0,width): 
            x_15[i,j]  = (x_15[i,j]-column_min[j])/(column_distance[j]+0.0000001)
    
    #x_15=preprocessing.scale(x_15)
    df_lable=pd.read_excel('151515lables.xlsx')
    height,width = df_lable.shape
    print(height,width,type(df_lable))
    y_15 = np.zeros((height,width))
    for i in range(0,height):
        for j in range(0,width): 
            y_15[i,j] = df_lable.iloc[i,j]
    print(y_15.shape)
    
    

    df_feature_21=pd.read_excel('21features_no.xlsx')
    height_21,width_21 = df_feature_21.shape
    x_21 = np.zeros([height_21,width_21])
    print(height_21,width_21,type(df_feature_21))
    for i_21 in range(height_21):
        for j_21 in range(width_21): 
            x_21[i_21,j_21]  = df_feature_21.iloc[i_21,j_21]
    print(x_21.shape)
    '''
    '''对x的每一个列项列进行预处理，使其每一列的值都在0-1之间'''
    '''
    column_max=x_21.max(axis=0)
    column_min=x_21.min(axis=0)
    column_distance=column_max-column_min
    
    
    
    for i in range(0,height_21):
        for j in range(0,width_21): 
            x_21[i,j]  = (x_21[i,j]-column_min[j])/(column_distance[j]+0.0000001)
    
    #x_21=preprocessing.scale(x_21)
    df_lable_21=pd.read_excel('21y_no.xlsx')
    height_21,width_21 = df_lable_21.shape
    print(height_21,width_21,type(df_lable_21))
    y_21 = np.zeros((height_21,width_21))
    for i in range(0,height_21):
        for j in range(0,width_21): 
            y_21[i,j] = df_lable_21.iloc[i,j]
    print(y_21.shape)

    '''




    
    '''这一部分的程序实现数据的过采样处理'''
    #shujuliang=len(x)
    #shujuliang_15=len(x_15)
    #shujuliang_21=len(x_21)
    #num=0
    
    
    #23846
    
    
    
    #jiluguzhangshuju=np.zeros((1000000,DIM))
    #for i in range(shujuliang):
        #if  y[i]==1:
            #jiluguzhangshuju[num]=x[i]
            #num=num+1
    #print('wwwwwooooooooooooooooooooooooo',num)
    #jiluguzhangshuju=jiluguzhangshuju[0:num]
    #s=smote.Smote(jiluguzhangshuju,N=1000)
    #aa=s.over_sampling()
    #print(aa.shape)
    #jilushuju=len(aa)
    #bb=np.ones((jilushuju,1))
    '''
    x_nochange=x
    y_nochange=y
    '''
    #x=np.r_[x,aa]
    #y=np.r_[y,bb]
    #print(x.shape)
    #print(y.shape)
   
    '''为了输出一部分的测试集数据'''
    '''
    S_X_train,S_X_test,S_Y_train,S_Y_test=cross_validation.train_test_split(x_nochange,y_nochange,test_size=0.25,random_state=5)
    '''
    #shurushijianchuangshujugeshu=len(x)
    #no_features,no_lables=time_window(x_nochange,y_nochange,48,12,shujuliang)
    #features_15,lables_15=time_window(x_15,y_15,48,12,shujuliang_15)
    #features_21,lables_21=time_window(x_21,y_21,48,12,shujuliang_21)
    #features,lables=time_window(x,y,48,12,shurushijianchuangshujugeshu)
    
    
    
    
    '''对数据进行降采样'''
    '''
    num_all=len(no_lables)
    
    num_down=0
    num_0=0
    num_1=1
    
    x_nochange_down=np.zeros((num_all,48,26))
    y_nochange_down=np.zeros((num_all,1))
    for t in range(num_all):
        if no_lables[t]==0:
            if (t%3==0):
                x_nochange_down[num_down]=no_features[t]
                y_nochange_down[num_down]=0
                num_down=num_down+1  
                num_0=num_0+1
                
        if no_lables[t]==1:
            x_nochange_down[num_down]=no_features[t]
            y_nochange_down[num_down]=1
            num_down=num_down+1
            num_1=num_1+1
            
    x_nochange_down_over=np.zeros((num_down,48,26))
    y_nochange_down_over=np.zeros((num_down,1))
    for tt in range(num_down):
        x_nochange_down_over[tt]=x_nochange_down[tt]
        y_nochange_down_over[tt]=y_nochange_down[tt]
    
    
    print('00000000000000000000000000',num_0)
    print('11111111111111111111111111',num_1)
    
    
    
    

    #print(features.shape)
    #print(lables.shape)
    '''
    '''这里加一个程序对标签Y 进行调整 使的标签是个两列的标签'''
    def Y_progress(Y):
        a=len(Y)
        wc=np.zeros([a,1])
        for i in range(a):
            if Y[i]==1:
                wc[i]=0
            if Y[i]==0:
                wc[i]=1
        return wc
    #wc=Y_progress(lables) 
    #no_wc=Y_progress(no_lables)
    #no_wc_down=Y_progress(y_nochange_down_over)
    #no_wc_15=Y_progress(lables_15)
    #no_wc_21=Y_progress(lables_21)
    #lables=np.concatenate((wc,lables),1)
    #no_lables=np.concatenate((no_wc,no_lables),1)
    #no_lables_down=np.concatenate((no_wc_down,y_nochange_down_over),1)
    #lables_15=np.concatenate((no_wc_15,lables_15),1)
    #lables_21=np.concatenate((no_wc_21,lables_21),1)
    
    train_y_wc=Y_progress(train_y)
    test_y_wc=Y_progress(test_y)
    train_y=np.concatenate((train_y_wc,train_y),1)
    test_y=np.concatenate((test_y_wc,test_y),1)
    
    
    
    
    #print('11111111111111111111111111',lables_21.shape)
    #print('5555555555555555555555555',features_21.shape)
    '''过采样数据'''
    #X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(features,lables,test_size=0.25,random_state=5) 
    '''降采样数据'''
    '''
    no_X_train_down,no_X_test_down,no_Y_train_down,no_Y_test_down=cross_validation.train_test_split(x_nochange_down_over,no_lables_down,
                                                                               test_size=0.8,random_state=5)
    '''
    '''正常的数据'''
    
    #no_X_train,no_X_test,no_Y_train,no_Y_test=cross_validation.train_test_split(no_features,no_lables,test_size=0.8,random_state=5) 
    
    #21 15 分别取 0.7 做训练剩下0.3做测试
    '''
    X_train_15,X_test_15,Y_train_15,Y_test_15=cross_validation.train_test_split(features_15,lables_15,test_size=0.3)
    X_train_21,X_test_21,Y_train_21,Y_test_21=cross_validation.train_test_split(features_21,lables_21,test_size=0.3)
    
    X_train=np.concatenate((X_train_15,X_train_21))
    X_test=np.concatenate((X_test_15,X_test_21))
    Y_train=np.concatenate((Y_train_15,Y_train_21))
    Y_test=np.concatenate((Y_test_15,Y_test_21))    
    '''
    
    #15做训练 21 做测试
    #X_train,X_test_15,Y_train,Y_test_15=cross_validation.train_test_split(features_15,lables_15,test_size=0.1)
    #X_train_21,X_test,Y_train_21,Y_test=cross_validation.train_test_split(features_21,lables_21,test_size=0.9)
    
    
    
    X_train,X_test_1,Y_train,Y_test_1=cross_validation.train_test_split(train_x,train_y,test_size=0.01)
    X_train_1,X_test,Y_train_1,Y_test=cross_validation.train_test_split(test_x,test_y,test_size=0.99)
    
    
    
    #a_csv=pd.DataFrame(no_lables)
    #a_csv.to_excel("no_lables13.xls")
    
    #print('xxxxxxxxxx的形状是',x.shape)
    '''x1=x[0:65534]
    a_csv=pd.DataFrame(x1)
    a_csv.to_excel("features_pred_1.xls")
    y1=y[0:65534]
    b_csv=pd.DataFrame(y1)
    b_csv.to_excel("lables_pred_1.xls")
    x2=x[65535:131068]
    a_csv=pd.DataFrame(x2)
    a_csv.to_excel("features_pred_2.xls")
    y2=y[65538:131068]
    b_csv=pd.DataFrame(y2)
    b_csv.to_excel("lables_pred_2.xls")
    
    x3=x[131070:196600]
    a_csv=pd.DataFrame(x3)
    a_csv.to_excel("features_pred_3.xls")
    y3=y[131070:196600]
    b_csv=pd.DataFrame(y3)
    b_csv.to_excel("lables_pred_3.xls")
    x4=x[196605:262100]
    
    a_csv=pd.DataFrame(x4)
    a_csv.to_excel("features_pred_4.xls")
    y4=y[196605:262100]
    
    b_csv=pd.DataFrame(y4)
    b_csv.to_excel("lables_pred_4.xls")
    
    x5=x[262140:327670]
    a_csv=pd.DataFrame(x5)
    a_csv.to_excel("features_pred_5.xls")
    y5=y[262140:327670]
    b_csv=pd.DataFrame(y5)
    b_csv.to_excel("lables_pred_5.xls")
    x6=x[327675:393200]
    a_csv=pd.DataFrame(x6)
    a_csv.to_excel("features_pred_6.xls")
    y6=y[327675:393200]
    b_csv=pd.DataFrame(y6)
    b_csv.to_excel("lables_pred_6.xls")
    
    x7=x[393210:458740]
    
    a_csv=pd.DataFrame(x7)
    a_csv.to_excel("features_pred_7.xls")
   
    y7=y[393210:458740]
    b_csv=pd.DataFrame(y7)
    b_csv.to_excel("lables_pred_7.xls")
    x8=x[458745:524200]
    a_csv=pd.DataFrame(x8)
    a_csv.to_excel("features_pred_8.xls")
    y8=y[458745:524200]
    b_csv=pd.DataFrame(y8)
    b_csv.to_excel("lables_pred_8.xls")
    x9=x[524280:589810]
    a_csv=pd.DataFrame(x9)
    a_csv.to_excel("features_pred_9.xls")
    y9=y[524280:589810]
    b_csv=pd.DataFrame(y9)
    b_csv.to_excel("lables_pred_9.xls")
    DATA_C=S_X_test[0:60000]
    c_csv=pd.DataFrame(DATA_C)
    c_csv.to_excel("no_features.xls")
    DATA_D=S_Y_test[0:60000]
    d_csv=pd.DataFrame(DATA_D)
    d_csv.to_excel("no_lables.xls")'''
    
    
    
    
    
    
    '''这里要将features，lables,no_features,no_lables 作为输出分别输出到不同的excel表格'''
    '''train_num=0
    test_num=0
    for i in range (38000):
        if Y_train[i,0]==1:
            train_num=train_num+1
    for j in range (12000):
        if Y_test[j,0]==1:
            test_num=test_num+1
    print("训练集中故障数据的个数大约是：",train_num)
    print("测试集当中故障数据的个数大约是：",test_num)'''            

#    return X_train,X_test,Y_train,Y_test,no_X_test,no_Y_test,no_X_train,no_Y_train,no_X_train_down,no_X_test_down,no_Y_train_down,no_Y_test_down
#    return no_X_test,no_Y_test,no_X_train,no_Y_train,no_X_train_down,no_X_test_down,no_Y_train_down,no_Y_test_down,X_train_21,X_test_21,Y_train_21,Y_test_21
    return X_train,X_test,Y_train,Y_test
    '''return X_train,X_test,Y_train,Y_test,no_X_train,no_Y_train,no_X_test,no_Y_test'''
    '''return X_train,X_test,Y_train,Y_test'''
''' features lables 就是要输入到CNN的数据'''

'''X_train,X_test,Y_train,Y_test=data_perprocessing()
print(X_test.shape)'''








