
from __future__ import print_function
import cPickle
import theano
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import random
import sys
import os
from PIL import Image
import numpy as np

def load_a_image(image_name):
    data = np.empty((1,1,28,28),dtype="float")
    img = Image.open(image_name)
    arr = np.asarray(img, dtype= "float")
    data[0,:,:,:] = arr
    return data

if __name__ == "__main__":
    origin_model = cPickle.load(open("model.pkl","rb"))
    number = len(sys.argv)
    for i in range(1, number):
        #load data
        data = load_a_image(sys.argv[i])
        pred_testlabel = origin_model.predict_classes(data,batch_size=1, verbose=1)
        print("[File No.",i,"]", sys.argv[i], "is", pred_testlabel)

