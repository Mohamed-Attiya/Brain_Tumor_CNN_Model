# Import Libraries & Packages :
import os
import itertools
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid') 

import warnings
warnings.filterwarnings('ignore')  

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization

# Data Preprocessing
data_dir = '/kaggle/input/brain-tumor-mri-images-44c' 
file_paths = []
labels = []
folds = os.listdir(data_dir)

for fold in folds:
    sub_fold_path = os.path.join(data_dir, fold)
    sub_folds = os.listdir(sub_fold_path)
    for sub_fold in sub_folds:
        sub_fold_paths = os.path.join(sub_fold_path, sub_fold)
        file_paths.append(sub_fold_paths)
        labels.append(fold.split(' ')[0])

F_Series = pd.Series(file_paths, name='file_paths')      
L_Series = pd.Series(labels, name='labels')
data_frame = pd.concat([F_Series, L_Series], axis=1)       

# Splitting Data Into (train, valid, test)
train_data_frame, dummy_data_frame = train_test_split(data_frame, train_size=0.8, random_state=42, stratify=data_frame['labels'])
valid_data_frame, test_data_frame = train_test_split(dummy_data_frame, train_size=0.5, random_state=42, stratify=dummy_data_frame['labels'])

# Create Image Data Generator :
batch_size = 16
img_size = (224, 224)
img_shape = (img_size[0], img_size[1], 3)
img_data_gen = ImageDataGenerator() 
train_data_gen = img_data_gen.flow_from_dataframe(train_data_frame, x_col='file_paths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
valid_data_gen = img_data_gen.flow_from_dataframe(valid_data_frame, x_col='file_paths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
test_data_gen = img_data_gen.flow_from_dataframe(test_data_frame, x_col='file_paths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

# Model Structure :    
base_model = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet', input_shape=img_shape)
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),  # Added BatchNormalization
    Dropout(rate=0.5),  
    Dense(len(train_data_gen.class_indices), activation='softmax')  
])

model.compile(optimizer=Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])  # Decreased learning rate
model.summary()  

# Model Training:
hist = model.fit(train_data_gen, epochs=20, verbose=1, validation_data=valid_data_gen)  # Increased epochs

# Plotting training history :
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], 'r', label='Training Loss')
plt.plot(hist.history['val_loss'], 'g', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.history['accuracy'], 'r', label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], 'g', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

train_score = model.evaluate(train_data_gen)
valid_score = model.evaluate(valid_data_gen)
test_score = model.evaluate(test_data_gen)

# Predictions :
preds = model.predict(test_data_gen)
y_pred = np.argmax(preds, axis=1)  
gin_dict = test_data_gen.class_indices

# Confusion Matrix
cm = confusion_matrix(test_data_gen.classes, y_pred)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(gin_dict))
plt.xticks(tick_marks, list(gin_dict.keys()), rotation=45)
plt.yticks(tick_marks, list(gin_dict.keys()))
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()    

model.save('brain_tumor.h5')