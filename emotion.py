#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import keras


# In[3]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os


# In[4]:


import glob 
train_dir =  glob.glob('C:\\Users\\uvash\\Downloads\\dataset\\train\\*\\*.jpg')


# In[5]:


import glob
val_dir = glob.glob('C:\\Users\\uvash\\Downloads\\dataset\\test\\*\\*.jpg')


# In[6]:


val_dir


# In[7]:


train_dir


# In[8]:


train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)


# In[9]:


import os


# In[10]:


train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\uvash\\Downloads\\dataset\\train',
    target_size = (48,48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = 'categorical'
)


# In[11]:


val_generator = val_datagen.flow_from_directory(
    'C:\\Users\\uvash\\Downloads\\dataset\\test',
    target_size = (48,48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = 'categorical'
)


# In[12]:


emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = (48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))


# In[13]:


emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])


# In[16]:


emotion_model_info = emotion_model.fit_generator(
  train_generator,
  validation_data=val_generator,
  epochs=20,
  steps_per_epoch=len(train_generator),
  validation_steps=len(val_generator)
)


# In[17]:


pip install torchvision


# In[18]:


import matplotlib.pyplot as plt


# In[20]:


# plot the loss
plt.plot(emotion_model_info.history['loss'], label='train loss')
plt.plot(emotion_model_info.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(emotion_model_info.history['accuracy'], label='train acc')
plt.plot(emotion_model_info.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[21]:


emotion_model.save_weights('model.h5')


# In[ ]:




