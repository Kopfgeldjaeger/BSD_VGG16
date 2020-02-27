import keras
import glob
import os
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,  array_to_img
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Dropout
from keras import optimizers
from keras.models import Sequential
import numpy as np

image_size=(224,224)

train_files = glob.glob('training-data/*')
train_imgs = [img_to_array(load_img(img, target_size=image_size)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [fn.split('\\')[1].strip()[0:1] for fn in train_files]

validation_files = glob.glob('validation-data/*')
validation_imgs = [img_to_array(load_img(img, target_size=image_size)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [fn.split('\\')[1].strip()[0:1] for fn in validation_files]

print('Train dataset shape:', train_imgs.shape, 
      '\tValidation dataset shape:', validation_imgs.shape)
"""
train_imgs = train_imgs.astype('float32')
validation_imgs = validation_imgs.astype('float32')
"""

num_classes = 4
epochs = 50
input_shape = (224, 224,3)

# encode text category labels
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

train_labels_array = np.array(train_labels)
# integer encode
le = LabelEncoder()
train_integer_encoded = le.fit_transform(train_labels_array)

# binary encode
ohe = OneHotEncoder(sparse=False)
train_integer_encoded = train_integer_encoded.reshape(len(train_integer_encoded), 1)
train_labels_ohe = ohe.fit_transform(train_integer_encoded)

# invert first example
#inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
validation_labels_array = np.array(validation_labels)

validation_integer_encoded = le.fit_transform(validation_labels_array)

# binary encode
ohe = OneHotEncoder(sparse=False)
validation_integer_encoded = validation_integer_encoded.reshape(len(validation_integer_encoded), 1)
validation_labels_ohe = ohe.fit_transform(validation_integer_encoded)




trdata = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='nearest')
traindata = trdata.flow(train_imgs,train_labels_ohe,batch_size=32)
valdata = ImageDataGenerator(rescale=1./255)
validationdata = valdata.flow(validation_imgs ,validation_labels_ohe,batch_size=8)

model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])



model.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5",monitor='val_acc',verbose=1,save_best_only=True,
                             save_weights_only= False,
                             mode ='auto',period=1)
early = EarlyStopping(monitor='val_acc',min_delta=0,patience=20,verbose=1,mode='auto')

             
history = model.fit_generator(traindata, steps_per_epoch=100, epochs=50,
                              validation_data=validationdata, validation_steps=50, 
                              callbacks=[checkpoint,early]
                              
                              )  
import matplotlib.pyplot as plt
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.plot(history.history["loss"]) 
plt.plot(history.history["val_loss"]) 
plt.title("model ACC")
plt.ylabel("ACC")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation loss"])
plt.show()
model.save('VGG16_multiclasses_model.h5')
