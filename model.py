from sklearn.metrics import accuracy_score
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, DepthwiseConv2D, GlobalAveragePooling2D, Input
import cv2
import numpy as np
import config
from keras.models import load_model,Sequential
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from make_data import make_data
from sklearn.model_selection import train_test_split

class my_model():
    def __init__(self,trainable):
        # các thông số cho quá trình train
        self.batch_size = config.BATCH_SIZE
        self.num_epochs = config.EPOCHS
        self.classes = config.N_CLASSES

        #kiểm tra nếu model chưa train thì bắt đầu train
        if trainable :
            self.build_model()
            self.model.summary()
            self.train()
        #nếu model đã train rồi thì load weights đã lưu ra để predict
        else :
            self.model = load_model('models_save/my_model.h5')

    def build_model(self):
        # CNN model
        '''' để có thể sử dụng API này trong keras thì input shape phải là 4 chiều '''
        self.model = Sequential()
        #thêm convolution layer
        self.model.add(Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=(96,96,3)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(3,3)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(1048,activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(self.classes,activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(1e-4), metrics=['accuracy'])
        
    


    def train(self):
        #data agumentation
        aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")
        # Model Checkpoint
        cpt_save = ModelCheckpoint('models_save/my_weight.h5', save_best_only=True, monitor='val_acc', mode='max')

        X_data,y_data = make_data()
        (X_train, X_val, y_train, y_val) = train_test_split(X_data, y_data, test_size=0.2, random_state=123)
        print("Training......")
        self.model.fit(aug.flow(X_train, y_train,batch_size=32), validation_data=(X_val, y_val), callbacks=[cpt_save], verbose=1, epochs=50, steps_per_epoch=len(X_train)/32)
        #lưu model sau khi đã trên xong
        self.model.save('models_save/my_model.h5')


    def predict(self,img):
        y_predict = self.model.predict(img.reshape(1,96,96,3))
        a = np.argmax(y_predict[0][:3])
        b = np.argmax(y_predict[0][3:])
        print(a)
        print(b)
        print('Giá trị dự đoán: ', y_predict[0])

