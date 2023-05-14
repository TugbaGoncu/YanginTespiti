import glob
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix,classification_report

### HILAL_KELES ###

# Bu kısım train-test-validation oluşturmak için sadece bir kez
# çalıştırılmalı (split-folders kütüphanesi gerekli)
# Yüklenmesi gereken kütüphane
# pip install split-folders

import splitfolders
# Buraya veriseti klasör yolu girilmeli
dataset_path = "dataset"

# Ilk parametre veriseti klasör yolu, ikinci parametre çıktı klasör yolu
# seed - rastgele bölme sayısı (farklı bilgisayarlarda aynı
# şekilde bölebilmek için), ratio - train, validation, test bölme oranı
splitfolders.ratio(dataset_path, output = "dataset_out", seed=1337, ratio=(.8, 0.1,0.1)) 

# Eğitim, test ve validasyon klaösr yolları
training_path = "dataset_out/train"
test_path = "dataset_out/test"
val_path = "dataset_out/val"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation ve generator (klasörlerden)
train_augmentation = ImageDataGenerator(
                                        rescale = 1. / 255)
train_generator = train_augmentation.flow_from_directory(
                                        directory = training_path,
                                        target_size = (224, 224),
                                        batch_size = 16)
val_augmenatation = ImageDataGenerator(
                                        rescale = 1. / 255)
val_generator = val_augmenatation.flow_from_directory(
                                        directory = val_path,
                                        target_size = (224, 224),
                                        batch_size = 1)
test_augmenatation = ImageDataGenerator(
                                        rescale = 1. / 255)
test_generator = test_augmenatation.flow_from_directory(
                                        directory = test_path,
                                        target_size = (224, 224),
                                        batch_size = 1)

# Model ve eğitim
from tensorflow.keras.applications.vgg16 import VGG16
model = VGG16(weights = None, classes=2)
model.summary()
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
opt = Adam(lr=0.001)
model.compile(optimizer = opt, loss = categorical_crossentropy, metrics=['accuracy'])
model.fit_generator(train_generator, epochs = 50, steps_per_epoch = int(3044 / 16),
                    validation_data = val_generator, validation_steps = 389,
                    verbose = 1)

### HILAL_KELES ###

lst_fire_img = glob.glob('C:/Users/Tuğba Göncü/bttrme/fire_dataset/fire_images/*.png')
lst_non_fire_img = glob.glob('C:/Users/Tuğba Göncü/bttrme/fire_dataset/non_fire_images/*.png')
lst_non_fire_img
print('Number of images with fire : {}'.format(len(lst_fire_img)))
print('Number of images with fire : {}'.format(len(lst_non_fire_img)))
lst_images_random = random.sample(lst_fire_img,10) + random.sample(lst_non_fire_img,10)
random.shuffle(lst_images_random)

plt.figure(figsize = (20,20))

for i in range(len(lst_images_random)):
    
    plt.subplot(4,5,i+1)


    if "non_fire" in lst_images_random[i]:
        img = cv2.imread(lst_images_random[i])
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        plt.imshow(img,cmap = 'gray')
        plt.title('Image without fire')

    else:
        img = cv2.imread(lst_images_random[i])
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        plt.imshow(img,cmap = 'gray')
        plt.title("Image with fire")



plt.show()
lst_fire = []
for x in lst_fire_img:
  lst_fire.append([x,1])
lst_nn_fire = []
for x in lst_non_fire_img:
  lst_nn_fire.append([x,0])
lst_complete = lst_fire + lst_nn_fire
random.shuffle(lst_complete)
df = pd.DataFrame(lst_complete,columns = ['files','target'])
df.head(10)
filepath_img = 'C:/Users/Tuğba Göncü/bttrme/fire_dataset/non_fire_images.189.png'
df = df.loc[~(df.loc[:,'files'] == filepath_img),:]
df.shape
plt.figure(figsize = (10,10))


sns.countplot(x = "target",data = df)

plt.show()
def preprocessing_image(filepath):
  img = cv2.imread(filepath) #read
  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #convert
  img = cv2.resize(img,(196,196))  # resize
  img = img / 255 #scale
  return img 
def create_format_dataset(dataframe):
  X = []
  y = []
  for f,t in dataframe.values:
    X.append(preprocessing_image(f))
    y.append(t)
  
  return np.array(X),np.array(y)
X, y = create_format_dataset(df)
X.shape,y.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify = y)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
model = Sequential()

model.add(Conv2D(128,(2,2),input_shape = (196,196,3),activation='relu'))
model.add(Conv2D(64,(2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32,(2,2),activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(1,activation= "sigmoid"))
model.summary()
callbacks = [EarlyStopping(monitor = 'val_loss',patience = 10,restore_best_weights=True)]
model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs = 30,batch_size = 32,callbacks = callbacks)
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(-1)
y_pred[y_pred<0.5] = 0
y_pred[y_pred>=0.5] = 1
y_pred = y_pred.astype('int')
y_pred
plt.figure(figsize = (20,10))

sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()
print(classification_report(y_test,y_pred))