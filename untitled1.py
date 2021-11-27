# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:28:13 2021

@author: aaq
"""

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPool2D, UpSampling2D,Dropout, Conv1D
from sklearn.utils import class_weight
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn.metrics as metrics
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.optimizers import SGD, Adadelta

#load dataset (288,22,750) 288=trial,22=channel,750=sample
num_classes = 4
X_1 = scipy.io.loadmat('E:/Masters/project/eeg_signal/1584.750/two_dimentional/subject4.mat',variable_names='class_1').get('class_1')
X_2=scipy.io.loadmat('E:/Masters/project/eeg_signal/1584.750/two_dimentional/subject4.mat',variable_names='class_2').get('class_2')
X_3=scipy.io.loadmat('E:/Masters/project/eeg_signal/1584.750/two_dimentional/subject4.mat',variable_names='class_3').get('class_3')
X_4=scipy.io.loadmat('E:/Masters/project/eeg_signal/1584.750/two_dimentional/subject4.mat',variable_names='class_4').get('class_4')
X=np.concatenate((X_1,X_2,X_3,X_4))
Y=np.zeros((288,))
Y[0:72]=1
Y[72:144]=2
Y[144:216]=3
Y[216:288]=4
#Shuffle data 
print('Shuffling...')
X, Y = shuffle(X, Y)
#Split data between 80% Training and 20% Testing
print('Splitting...')
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, train_size=.8, test_size=.2, shuffle=True)
#disoin filter
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import butter, lfilter
  
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a
# filter test
sampRat = 250
dt = 1/sampRat
T = 10 
#t = np.linspace(0, T, T*sampRat, endpoint=False)
#y = 5*np.sin(2*math.pi*t*10)+10*np.sin(2*math.pi*t*35)
#filterY = butter_bandpass_filter(y, 8,30, sampRat, 4)
#f = np.linspace(0, sampRat, T*sampRat, endpoint=False)
#ff = np.fft.fft(y)
#ff = np.abs(ff)*2/T/sampRat
#plt.figure()
#plt.plot(f, ff)
#plt.title('FFT before filter')
#plt.show()
 
#ff = np.fft.fft(filterY)
#ff = np.abs(ff)*2/T/sampRat
#plt.figure()
#plt.plot(f, ff)
#plt.title('FFT after filter')
#EEG FFT compare between filter and filter before
# FFT transfrom
sampleRate=250
import numpy as np
import pywt
for i in range(230):
    dataFtt = np.fft.fft(x_train[i,7,:])  # C3 channel
    Freq = np.linspace(1/x_train.shape[2],sampleRate,x_train.shape[2])
# plot FFT

    #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    #ax1.plot(Freq, np.abs(dataFtt), color='blue')
    #ax1.set_title('C3 Channel Filter Before')
    #plt.xlabel('Freq (in Hz)')
    #plt.ylabel('Amplitude')
    filterY = butter_bandpass_filter(x_train,8,20,sampleRate, 5)
    dataFtt = np.fft.fft(filterY[i,7,:])    # C3
    #ax2.plot(Freq, np.abs(dataFtt), color='green')
    #ax2.set_title('C3 Channel Filter After')

    Dataftt=np.fft.fft(x_train[i,11,:]) #c4 channel
    Freqencyc4 = np.linspace(1/x_train.shape[2],sampleRate,x_train.shape[2])
# plot FFT
   # fig, (AX1, AX2) = plt.subplots(2, 1, sharex=True)
   # AX1.plot(Freqencyc4, np.abs(Dataftt), color='blue')
    #AX1.set_title('C4 Channel Filter Before')
    #plt.xlabel('Freqencyc4 (in Hz)')
    #plt.ylabel('Amplitude')
    filterYc4 = butter_bandpass_filter(x_train,8,20,sampleRate, 5)
    Dataftt = np.fft.fft(filterYc4[i,11,:])    # C4
    #AX2.plot(Freqencyc4, np.abs(Dataftt), color='red')
    #AX2.set_title('C4 Channel Filter After')

#wavelet
    
    print(plt.style.available)
    plt.style.use('classic')
    wavlist = pywt.wavelist(kind='continuous')
    print("Class of continuous wavelet functions：")
    print(wavlist)
    sampling_rate = 250 
    t = np.arange(3,6.0,1.0/sampling_rate)
    wavename = "cmor3_3"
    totalscal = 64    # scale 
    fc = pywt.central_frequency(wavename) #  central frequency
    cparam = 2 * fc * totalscal
    scales = cparam/np.arange(1,totalscal+1)
# C3 channel
    [cwtmatr3, frequencies3] = pywt.cwt(filterY[i,7,:],scales,wavename,1.0/sampling_rate) # continuous wavelet transform
    fig = plt.figure(1)
    #plt.contourf(t, frequencies3, abs(cwtmatr3))
    #plt.ylabel(u"freq(Hz)")
    #plt.xlabel(u"time(s)")
    #plt.colorbar()
    #fig.savefig('C3.png')
    #fig = plt.figure(2)
# C4 channel
    [cwtmatr4, frequencies4] = pywt.cwt(filterYc4[i,11,:],scales,wavename,1.0/sampling_rate) 
   # plt.contourf(t, frequencies4, abs(cwtmatr4))
    #plt.ylabel(u"freq(Hz)")
    #plt.xlabel(u"time(s)")
    #plt.colorbar()
    #fig.savefig('C4.png')
    cwtmatr = np.concatenate([abs(cwtmatr3[0:30,:]), abs(cwtmatr4[0:30,:])],axis=0) 
    fig = plt.figure()
    plt.contourf(cwtmatr)
    plt.xticks([])  # remove x
    plt.yticks([])  # remove y
    plt.axis('off') # remove axis
    fig.set_size_inches(800/100.0,600/100.0)#  set pixels width*height
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0) 
    plt.margins(0,0)
    figureName = str(i)
    if y_train[i] == 1:
       filepath = r'E:\Masters\project\eeg_signal\1584.750\two_dimentional\Wavelet\sub4\wt_train\1\{}.png'.format(figureName)
       fig.savefig(filepath)
    if y_train[i]== 2:
       filepath = r'E:\Masters\project\eeg_signal\1584.750\two_dimentional\Wavelet\sub4\wt_train\2\{}.png'.format(figureName)
       fig.savefig(filepath)
    if y_train[i]== 3:
       filepath = r'E:\Masters\project\eeg_signal\1584.750\two_dimentional\Wavelet\sub4\wt_train\3\{}.png'.format(figureName)
       fig.savefig(filepath)
    if y_train[i]== 4:
       filepath = r'E:\Masters\project\eeg_signal\1584.750\two_dimentional\Wavelet\sub4\wt_train\4\{}.png'.format(figureName)
       fig.savefig(filepath)
print('wavelet transfrom completed')

####datatest
for j in range(58):
    dataFtt = np.fft.fft(x_test[j,7,:])  # C3 channel
    Freq = np.linspace(1/x_test.shape[2],sampleRate,x_test.shape[2])
# plot FFT

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    #ax1.plot(Freq, np.abs(dataFtt), color='blue')
    #ax1.set_title('C3 Channel Filter Before')
    #plt.xlabel('Freq (in Hz)')
    #plt.ylabel('Amplitude')
    filterY = butter_bandpass_filter(x_test,8,20,sampleRate, 5)
    dataFtt = np.fft.fft(filterY[j,7,:])    # C3通道
    #ax2.plot(Freq, np.abs(dataFtt), color='green')
    #ax2.set_title('C3 Channel Filter After')

    Dataftt=np.fft.fft(x_test[j,11,:]) #c4
    Freqencyc4 = np.linspace(1/x_test.shape[2],sampleRate,x_test.shape[2])
# plot FFT
    fig, (AX1, AX2) = plt.subplots(2, 1, sharex=True)
    #AX1.plot(Freqencyc4, np.abs(dataFtt), color='blue')
    #AX1.set_title('C4 Channel Filter Before')
    #plt.xlabel('Freqencyc4 (in Hz)')
    #plt.ylabel('Amplitude')
    filterYc4 = butter_bandpass_filter(x_test,8,20,sampleRate, 5)
    Dataftt = np.fft.fft(filterY[j,11,:])    # C4
    #AX2.plot(Freqencyc4, np.abs(Dataftt), color='red')
    #AX2.set_title('C4 Channel Filter After')

#wavelet
    
    print(plt.style.available)
    plt.style.use('classic')
    wavlist = pywt.wavelist(kind='continuous')
    print("Class of continuous wavelet functions：")
    print(wavlist)
    sampling_rate = 250 
    t = np.arange(3,6.0,1.0/sampling_rate)
    wavename = "cmor3_3"
    totalscal = 64    # scale 
    fc = pywt.central_frequency(wavename) #  central frequency
    cparam = 2 * fc * totalscal
    scales = cparam/np.arange(1,totalscal+1)
# C3 channel
    [cwtmatr3, frequencies3] = pywt.cwt(filterY[j,7,:],scales,wavename,1.0/sampling_rate) # continuous wavelet transform
    #fig = plt.figure(1)
    #plt.contourf(t, frequencies3, abs(cwtmatr3))
    #plt.ylabel(u"freq(Hz)")
    #plt.xlabel(u"time(s)")
    #plt.colorbar()
    #fig.savefig('C3.png')
    #fig = plt.figure(2)
# C4 channel
    [cwtmatr4, frequencies4] = pywt.cwt(filterYc4[j,11,:],scales,wavename,1.0/sampling_rate) 
    #plt.contourf(t, frequencies3, abs(cwtmatr3))
    #plt.ylabel(u"freq(Hz)")
    #plt.xlabel(u"time(s)")
    #plt.colorbar()
    #fig.savefig('C4.png')
    cwtmatr = np.concatenate([abs(cwtmatr3[0:30,:]), abs(cwtmatr4[0:30,:])],axis=0) 
    fig = plt.figure()
    plt.contourf(cwtmatr)
    plt.xticks([])  # remove x
    plt.yticks([])  # remove y
    plt.axis('off') # remove axis
    fig.set_size_inches(800/100.0,600/100.0)#  set pixels width*height
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0) 
    plt.margins(0,0)
    figureName = str(j)
    if y_test[j] == 1:
       filepath = r'E:\Masters\project\eeg_signal\1584.750\two_dimentional\Wavelet\sub4\wt_test\1\{}.png'.format(figureName)
       fig.savefig(filepath)
    if y_test[j]== 2:
       filepath = r'E:\Masters\project\eeg_signal\1584.750\two_dimentional\Wavelet\sub4\wt_test\2\{}.png'.format(figureName)
       fig.savefig(filepath)
    if y_test[j]== 3:
       filepath = r'E:\Masters\project\eeg_signal\1584.750\two_dimentional\Wavelet\sub4\wt_test\3\{}.png'.format(figureName)
       fig.savefig(filepath)
    if y_test[j]== 4:
       filepath = r'E:\Masters\project\eeg_signal\1584.750\two_dimentional\Wavelet\sub4\wt_test\4\{}.png'.format(figureName)
       fig.savefig(filepath)
print('wavelet transfrom completed')
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
def get_file(filename):
    ''' load time frequency diagram '''
    dataTrain = list()
    labelTrain = list()
    for label in os.listdir(filename):
        for pic in os.listdir(filename+label):
            dataTrain.append(filename+label+'/'+pic)  # shuffle data
            labelTrain.append(int(label))
    temp = np.array([dataTrain, labelTrain])          
    temp = np.transpose(temp)  
    np.random.shuffle(temp)
    image_list = temp[:,0]
    label_list = temp[:,1]
    label_list = [int(i) for i in label_list]  
    return image_list, label_list
pathname_train = r"E:/Masters/project/eeg_signal/1584.750/two_dimentional/Wavelet/sub1/wt_train/"
image_train_list, label_train_list = get_file(pathname_train)
# read train data  figure-->tensor
# read_file_train 
# decode_jpeg_train
X_train = np.empty([230,64,64,4])
with tf.Session() as sess:
    for i in range(len(image_train_list)):
        image_raw_data = tf.gfile.GFile(image_train_list[i],'rb').read() 
        image_data = tf.image.decode_jpeg(image_raw_data)
        resized = tf.image.resize_images(image_data, [64,64],method=0)  
        resized = np.asarray(resized.eval(),dtype='uint8')       
        X_train[i,:,:,:]=resized  
# train data
X_Train = X_train[:,:,:,:]
Y_train = np.subtract(np.array(label_train_list), 1)
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)
# convert integers to dummy variables (i.e. one hot encoded)
Y_Train = np_utils.to_categorical(encoded_Y)
#test
pathname_test = r"E:/Masters/project/eeg_signal/1584.750/two_dimentional/Wavelet/sub1/wt_test/"
image_test_list, label_test_list = get_file(pathname_test)
# read train data  figure-->tensor
# read_file 
# decode_jpeg
X_test = np.empty([58,64,64,4])
with tf.Session() as sess:
    for i in range(len(image_test_list)):
        image_raw_data = tf.gfile.GFile(image_test_list[i],'rb').read() 
        image_data = tf.image.decode_jpeg(image_raw_data)
        # resize
        resized = tf.image.resize_images(image_data, [64,64],method=0)  
        resized = np.asarray(resized.eval(),dtype='uint8')      
        X_test[i,:,:,:]=resized  
# test data
X_Test = X_test[:,:,:,:]
Y_test = np.subtract(np.array(label_test_list), 1)
encoder = LabelEncoder()
encoder.fit(Y_test)
encoded_Y = encoder.transform(Y_test)
# convert integers to dummy variables (i.e. one hot encoded)
Y_Test = np_utils.to_categorical(encoded_Y)
# resize
X_Train = X_Train/255
X_Test=X_Test/255.
print ("X_Train shape: " + str(X_Train.shape))
print ("X_Test shape: " + str(X_Test.shape))
print ("Y_Test shape: " + str(Y_Test.shape))
print ("Y_Train shape: " + str(Y_Train.shape))
#opt=keras.optimizers.Adam( lr=0.0003,epsilon=1e-07)
opt=SGD(lr=0.00001, momentum=0.99, decay=0.01)
input_shape = X_Train.shape[1:]
#opt=Adadelta(lr=0.001)
model = Sequential()
model.add(Conv2D(256,kernel_size=8,activation='relu', input_shape =input_shape))
model.add(MaxPool2D(pool_size=(4,4)))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics= ['accuracy'])
history=model.fit(X_Train, Y_Train,batch_size=8,validation_split = 0.2, epochs = 60, verbose = 1)

#score = model.evaluate(x_test, y_test,batch_size=32, verbose =1)
#print("\nTest Set Validation Results-> %s: %.2f%%" % (model.metrics_names[1], score[1]*100) ,"%s: %.2f%%" % (model.metrics_names[0], score[0]*100) )
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()