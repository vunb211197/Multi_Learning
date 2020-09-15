import os
import cv2
import numpy as np
from sklearn.utils import shuffle

labels={'black':0,'blue':1,'red':2,'jeans':3,'shirt':4,'dress':5}
data_path = 'dataset'


def processing_labesl(dir):
    s = []
    q= []
    a = str(dir).split("_")
    for j in a : 
        q.append(labels[j])
    for i in range(6):
        if q.__contains__(i):
            s.append(1)
        else :
            s.append(0)
    return np.asarray(s)



def make_data():
    X_data=[]
    y_data=[]
    for dir in os.listdir(data_path):
        class_dir = os.path.join(data_path,dir)
        for image in os.listdir(class_dir):
            img = cv2.imread(os.path.join(class_dir,image))
            img = cv2.resize(img,(96,96))
            # để model hội tụ nhanh hơn thì chia toàn bộ các pixel
            img = img/255.0
            X_data.append(img)
            y_data.append(processing_labesl(dir))

    #chuyển sang numpy array
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)

    X_data,y_data = shuffle(X_data,y_data)

    return X_data, y_data


