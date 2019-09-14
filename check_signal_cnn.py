'''
2019/09/12
使用cnn检测是否有心率问题
'''

import numpy as np
import pandas as pd
import cv2
import os
import random

import keras
from keras.layers import Input,Conv2D,MaxPool2D,Softmax,BatchNormalization,Reshape,Dense
from keras.models import load_model,save_model,Model
from keras.callbacks import ModelCheckpoint

train_path='E:\\AI_Project\\data_set\\data\\hf_round1_train\\image\\image2'
model_path='E:\\AI_Project\\tianchi_competition\\heart_signal\\model\\'
h,w=4000,5000

def load_img(file_path):
    img=cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    return img
def load_label():
    lab=pd.read_csv('E:\\AI_Project\\data_set\\data\\label.csv',encoding='utf-8')
    return lab
def get_train_valid_id_list():
    id_list = []
    for filename in os.listdir(train_path):
        id = filename.split('.')[0]
        id_list.append(id)

    L1=len(id_list)//10
    L2=len(id_list)-L1
    random.shuffle(id_list)
    id_list_train=id_list[0:L2]
    id_list_valid=id_list[L2:L1+L2]
    return id_list_train,id_list_valid

def generate_TrainData(batch_size,id_list=[],head=None):
    while True:
        train_data=[]
        train_label=[]
        batch=0
        lab=load_label()
        for i in range(len(id_list)):
            id=int(id_list[i])
            batch+=1
            img=load_img(train_path+'\\'+str(id)+'.png')
            label=lab[head][lab.id==id].values[0]
            train_data.append(load_img())
            train_label.append(label)

            if batch==batch_size:
                train_data=np.array(train_data)
                train_label=np.array(train_label)
                train_label.reshape((batch_size,1))

                yield (train_data,train_label)
                batch=0
                train_data = []
                train_label = []
def generate_ValidData(batch_size,id_list=[],head=None):
    while True:
        valid_data=[]
        valid_label=[]
        batch=0
        lab=load_label()
        for i in range(len(id_list)):
            id=int(id_list[i])
            batch+=1
            img=load_img(train_path+'\\'+str(id)+'.png')
            label=lab[head][lab.id==id].values[0]
            valid_data.append(load_img())
            valid_label.append(label)

            if batch==batch_size:
                valid_data=np.array(train_data)
                valid_label=np.array(train_label)
                valid_label.reshape((batch_size,1))

                yield (valid_data,valid_label)
                batch=0
                valid_data=[]
                valid_label=[]

def heart_signal_cnn():
    inputs=Input((2048,2048))
    s=Reshape((2048,2048,1))(inputs)

    c1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(s)
    c1 = BatchNormalization()(c1)
    p1 = MaxPool2D((2, 2))(c1)
    #(none,1024,1024,32)
    c2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPool2D((2, 2))(c2)
    # (none,512,512,64)
    c3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPool2D(2, 2)(c3)
    # (none,256,256,128)
    c4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPool2D((2, 2))(c4)
    # (none,128,128,256)
    c5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c5)
    c5 = BatchNormalization()(c5)
    p5 = MaxPool2D((2, 2))(c5)
    # (none,64,64,512)
    c6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(p5)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c6)
    c6 = BatchNormalization()(c6)
    p6 = MaxPool2D((2, 2))(c6)
    # (none,32,32,512)
    c7 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(p6)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c7)
    c7 = BatchNormalization()(c7)
    p7 = MaxPool2D((2, 2))(c7)
    # (none,16,16,512)
    c8 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(p7)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c8)
    c8 = BatchNormalization()(c8)
    p8 = MaxPool2D((2, 2))(c8)
    # (none,8,8,512)
    d1 = Reshape((8 * 8 * 512,))(p8)
    # (none,8*8*512)
    d1 = Dense(units=120, activation='relu', use_bias=True)(d1)
    d1 = BatchNormalization()(d1)
    # (none,120)
    d2 = Dense(units=64, activation='relu', use_bias=True)(d1)
    d2 = BatchNormalization()(d2)
    # (none,64)
    d3 = Dense(units=2, activation='softmax', use_bias=True)(d2)

    outputs = d3

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model
def train():
    EPOCHS = 30
    # BS=16
    BS = 10  #####可修改
    model = heart_signal_cnn()
    modelcheck = ModelCheckpoint(modelpath + 'hs_cnn_1.h5', monitor='val_acc', save_best_only=True,mode='max')  #####修改模型名称
    # 监控模型中的'val_acc'参数，当该参数增大时，保存模型（保存最佳模型）
    callable = [modelcheck]  # 回调函数
    id_list_train,id_list_valid=get_train_valid_id_list()
    train_numb = len(id_list_train)
    valid_numb = len(id_list_valid)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)
    H = model.fit_generator(generator=generateTrainData(BS,id_list_train),  # 一个generator或Sequence实例
                            steps_per_epoch=train_numb // BS,  # 从generator产生的步骤的总数（样本批次总数）。通常情况下，应该等于数据集的样本数量除以批量的大小。
                            epochs=EPOCHS,  # 整数，在数据集上迭代的总数。
                            verbose=1,  # 日志显示,0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                            validation_data=generateValidData(BS,id_list_valid),  # 生成验证集的生成器
                            validation_steps=valid_numb//BS,
                            callbacks=callable,
                            max_queue_size=10)  # 生成器队列的最大容量
    # 分批训练,生成器，返回一个history


def test():
    # img=load_img(train_path+'\\'+'100.png')
    # print(img.shape)
    # cv2.imshow('img',img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # lab=load_label()
    # a=lab[lab.id==14].index.tolist()
    # print(lab)
    # print(a)

    # lab = load_label()
    # a = lab['age'][lab.id == 14].values[0]
    # print(lab)
    # print(a)

    # id_list = []
    # for filename in os.listdir(train_path):
    #     id = filename.split('.')[0]
    #     id_list.append(id)
    # l1,l2=randen_split(id_list)
    # print(l1)
    # print(l2)

    l1,l2=get_train_valid_id_list()
    print(l1)
    print(l2)

if __name__ == '__main__':
    # test()
    # load_label()
    # heart_signal_cnn()
    train()