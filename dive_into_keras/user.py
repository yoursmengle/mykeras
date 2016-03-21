'''
Author:wepon
Code:https://github.com/wepe

File: cnn-svm.py
'''
from __future__ import print_function
import cPickle
import theano
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from data import load_data
import random
from userdata import load_userdata
import sys
import os
from PIL import Image
import numpy as np

def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)
    
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)

def rf(traindata,trainlabel,testdata,testlabel):
    print("Start training Random Forest...")
    rfClf = RandomForestClassifier(n_estimators=400,criterion='gini')
    rfClf.fit(traindata,trainlabel)
    
    pred_testlabel = rfClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-rf Accuracy:",accuracy)

def load_a_image(image_name):
    data = np.empty((1,1,28,28),dtype="float32")
    img = Image.open(image_name)
    arr = np.asarray(img, dtype= "float32")
    data[0,:,:,:] = arr
    
    return data

if __name__ == "__main__":
    origin_model = cPickle.load(open("model.pkl","rb"))
    number = len(sys.argv)
    for i in range(1, number):
        #load data
        data = load_a_image(sys.argv[i])
        pred_testlabel = origin_model.predict_classes(data,batch_size=1, verbose=1)
        print(sys.argv[i], "is", pred_testlabel)

