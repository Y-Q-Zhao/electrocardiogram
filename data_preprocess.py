'''
2019/09/10
2019/09/11
2019/09/12
对心电图数据进行预处理
'''
import numpy as np
import pandas as pd
import glob
# import seaborn as sns
import matplotlib.pyplot as plt
import os

path_train=    'E:\\AI_Project\\data_set\\data\\hf_round1_train\\train\\'
# path_train_img='E:\\AI_Project\\data_set\\data\\hf_round1_train\\image\\image1'
path_train_img='E:\\AI_Project\\data_set\\data\\hf_round1_train\\image\\image2'
path_test=     'E:\\AI_Project\\data_set\\data\\hf_round1_testA\\testA\\'
path_label=    'E:\\AI_Project\\data_set\\data\\hf_round1_label.txt'
path_sub=      'E:\\AI_Project\\data_set\\data\\hf_round1_subA.txt'
path_arrythmia='E:\\AI_Project\\data_set\\data\\hf_round1_arrythmia.txt'
#-----------------------------------------------------------------------------------------------------------------------
def read_label():
    lab=pd.read_csv(path_label,sep='.txt',encoding='utf-8',engine='python',header=None)

    df=pd.DataFrame()
    df['id']=lab[0]
    info=[]
    for x in lab[1].values:
        str=x.split('\t')
        info.append([str[1],str[2],str[3:]])
    info=np.array(info)
    df['age']=info[:,0]
    df['gender']=info[:,1]
    df['arrythmia']=info[:,2]

    h,w=df.shape
    for i in range(h):
        if df['age'][i]=='':
            df['age'][i]=-1
        else:
            df['age'][i]=int(df['age'][i])
        if df['gender'][i]=='FEMALE':
            df['gender'][i]=0
        elif df['gender'][i]=='MALE':
            df['gender'][i]=1
        else:
            df['gender'][i]=-1

    return df
def read_subA():
    subA=pd.read_csv(path_sub,sep='.txt', encoding='utf-8', engine='python', header=None)

    df = pd.DataFrame()
    df['id'] = subA[0]
    info = []
    for x in subA[1].values:
        try:
            str = x.split('\t')
            info.append([str[1], str[2]])
        except:
            info.append(['',''])
    info = np.array(info)
    df['age'] = info[:, 0]
    df['gender'] = info[:, 1]

    h, w = df.shape
    for i in range(h):
        if df['age'][i] == '':
            df['age'][i] = -1
        else:
            df['age'][i] = int(df['age'][i])
        if df['gender'][i]=='FEMALE':
            df['gender'][i]=0
        elif df['gender'][i]=='MALE':
            df['gender'][i]=1
        else:
            df['gender'][i]=-1

    print(df)
    return df
def read_train(file_path):
    '''
    根据公式，将心电数据从8联转换成12联
    '''
    df=pd.read_csv(file_path,sep=' ')
    df['III']=df['II']-df['I']
    df['aVR']=-(df['I']+df['II'])/2
    df['aVL']=df['I']-df['II']/2
    df['aVF']=df['II']-df['I']/2
    return df
def read_arrythmia():
    arrythmia=[]
    for line in open(path_arrythmia,'r',encoding='utf-8'):
        arrythmia.append(line.split('\n')[0])
    # print(arrythmia)
    # print(len(arrythmia))
    return arrythmia
#-----------------------------------------------------------------------------------------------------------------------
def make_file():
    '''
    按照不同部位的心电图，创建不同名字的文件
    :return: none
    '''
    heads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL', 'aVF']
    for head in heads:
        os.makedirs(path_train_img+'\\'+head)
def get_filename(filepath):
    filename=filepath.split('\\')[-1]
    filename=filename.split('.')[0]
    return filename
#-----------------------------------------------------------------------------------------------------------------------
def draw_image(df,head,filepath):
    '''
    画图，生成图像并保存
    :param df: 需要画图的数据
    :param head: 需要画图数据在dataframe中的表头
    :param filepath: 保存图像的路径
    :return: none
    '''
    size_h,size_w=10.24,20.48

    plt.figure(figsize=(size_w,size_h))
    plt.plot(np.array(df[head].index), df[head].values)
    plt.axis('off')
    plt.gcf().set_size_inches(size_w,size_h)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(filepath)
    plt.close()
def draw_image_total(df,filepath):
    '''
    画图，生成图像并保存
    :param df: 需要画图的数据
    :param filepath: 保存图像的路径
    :return: none
    '''
    # df=read_train('E:\\AI_Project\\data_set\\data\\hf_round1_train\\train\\2.txt')
    plt.figure()

    for i, item in enumerate(df.columns):
        plt.subplot(len(df.columns), 1, i + 1)
        plt.plot(np.array(df[item].index), df[item].values)

        plt.axis('off')
        plt.gcf().set_size_inches(20.48,20.48)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
    # plt.show()
    plt.savefig(filepath)
    plt.close()
def train_data_visualize1():
    heads=['I','II','V1','V2','V3','V4','V5','V6','III','aVR','aVL','aVF']
    file_list=os.listdir(path_train)
    for file_path in file_list:
        data=read_train(path_train+file_path)
        filename=get_filename(file_path)
        for head in heads:
            filepath=path_train_img+'\\'+head+'\\'+filename+'.png'
            draw_image(data,head,filepath)
        print(filename,'is done!')
def train_data_visualize2():
    heads=['I','II','V1','V2','V3','V4','V5','V6','III','aVR','aVL','aVF']
    file_list=os.listdir(path_train)
    for file_path in file_list:
        data=read_train(path_train+file_path)
        filename=get_filename(file_path)
        filepath=path_train_img+'\\'+filename+'.png'
        draw_image_total(data,filepath)
        print(filename,'is done!')
def transform_label():
    '''
    将label转换成one-hot格式，并保存成csv格
    :return:
    '''
    arrythmia=read_arrythmia()
    label=read_label()
    (h,w)=label.shape
    for a in arrythmia:
        colum=[]
        for i in range(h):
            if a not in label['arrythmia'][i]:
                colum.append(0)
            else:
                colum.append(1)
        # label[a]=np.array(colum)
        label[a]=colum

    label.to_csv('E:\\AI_Project\\data_set\\data\\label.csv',encoding='utf-8-sig',index=None)
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # train_data_visualize1()
    # train_data_visualize2()
    draw_image_total()
    # read_arrythmia()
    # transform_label()
    # test()