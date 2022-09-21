import numpy as np
import Dimension_reduction as dr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xlwt
import pydot
from sklearn.metrics import median_absolute_error
import tensorflow as tf
from tensorflow.keras import Sequential,layers,utils,losses
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


#构造数据集
def create_dataset(x,y,seq_len=10):#滑窗为10
    features=[]
    targets=[]
    for i in range(0,len(x)-seq_len,1):
        data=x.iloc[i:i+seq_len] #序列数据
        label=y.iloc[i+seq_len] #标签数据
        #保存到features和labels
        features.append(data)
        targets.append(label)

    return np.array(features),np.array(targets)

#构造批数据
def create_batch_dataset(x,y,train=True,buffer_size=1000,batch_size=128):#128个窗口，每1000个窗口做一次打乱
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y))) #数据封装，tensor类型
    if train: #训练集
        #cache存入内存，加速读取，shuffle打乱窗口，batch构建批数据
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else: #测试集
        return batch_data.batch(batch_size)


# 划分训练集和测试集
def split(data):

    x = data.drop(["期货收盘价(活跃合约):精对苯二甲酸(PTA)", "期货收盘价(活跃合约):乙二醇", "期货收盘价(活跃合约):短纤"], axis="columns")
    y = data[["期货收盘价(活跃合约):精对苯二甲酸(PTA)", "期货收盘价(活跃合约):乙二醇", "期货收盘价(活跃合约):短纤"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False,random_state=66)
    # return y_train
    #构造训练集
    train_dataset,train_labels=create_dataset(x_train,y_train,seq_len=10)
    # print(train_dataset.shape)
    # print(train_labels.shape)
    test_dataset, test_labels = create_dataset(x_test,y_test,seq_len=10)
    # print(test_dataset.shape)
    # print(test_labels.shape)
    #训练集批数据
    train_batch_dataset=create_batch_dataset(train_dataset,train_labels)
    #测试集批数据
    test_batch_dataset=create_batch_dataset(test_dataset,test_labels,train=False)
    #从测试批数据中，获取一个batch_size的样本数据 128个窗口
    # print(list(test_batch_dataset.as_numpy_iterator())[0])
    return train_batch_dataset,test_batch_dataset

def LSTM(train_batch_dataset,test_batch_dataset):
    #存储批处理数据
    train_batch_dataset=train_batch_dataset
    test_batch_dataset=test_batch_dataset
    #模型搭建
    model = Sequential((
        layers.LSTM(units=20,input_shape=(10,61),return_sequences=True),#True表示状态向后传播
        layers.Dropout(0.4),
        layers.LSTM(units=20,return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(units=16,return_sequences=True),
        layers.LSTM(units=10),
        layers.Dense(1)
    ))
    #显示模型结构
    print(utils.plot_model(model))
    #模型编译
    model.compile(optimizer='adam',loss='mse')
    #保存最佳模型
    checkpoint_file = "best_model.hdf5"
    checkpoint_callback=ModelCheckpoint(filepath=checkpoint_file,
                                        monitor='loss',
                                        mode='min',
                                        save_best_only=True,
                                        save_weights_only=True)
    #模型训练
    history=model.fit(train_batch_dataset,
                      epochs=10,
                      validation_data=test_batch_dataset,
                      callbacks=[checkpoint_callback])
    print(history)
    #显示训练结果
    history.history['val_loss'] = [i / 10 for i in history.history['val_loss']]
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'],label='train loss')
    plt.plot(history.history['val_loss'],label='val loss')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    df = dr.load_data()
    train_batch_dataset,test_batch_dataset=split(df)
    LSTM(train_batch_dataset,test_batch_dataset)