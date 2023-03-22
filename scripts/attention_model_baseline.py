import os
import gc
import shutil
import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, auc

from attention_layer import AttentionLayer
from utils import my_precision, my_recall, my_f1, my_roc_auc, get_balanced_acc, plotConfusionMatrix, plotCurves, plotROCCurve

## Empty /kaggle/working + Free memory usage
folder = './'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

tf.keras.backend.clear_session()
gc.collect()

# Load data
data_dir = '/kaggle/input/coughvid-melspectrum-extracted/coughvid_melspec'

def load_set(data_dir, set_name):
    path = os.path.join(data_dir, f'{set_name}_coughvid_melspec.npz')
    features = np.load(path)
    X = features['images']
    y = features['covid_status']
    
    return X, y
    
X_train, y_train = load_set(data_dir, 'train')
X_valid, y_valid = load_set(data_dir, 'valid')

# Format images according to data format required
cols, rows = 88, 39

def format1(images, rows, cols):
    return images.reshape(images.shape[0],3,rows,cols)

def format2(images, rows, cols):
    return images.reshape(images.shape[0],rows,cols,3)

if K.image_data_format() == 'channels_first':
    X_train = format1(X_train, rows, cols)
    X_valid = format1(X_valid, rows, cols)
    
    input_shape = (3, rows, cols)
else:
    X_train = format2(X_train, rows, cols)
    X_valid = format2(X_valid, rows, cols)
    
    input_shape = (rows, cols,3)

## Evaluation metrics
METRICS = [
    tensorflow.keras.metrics.BinaryAccuracy(name='accuracy'),
    tensorflow.keras.metrics.Precision(name='precision'),
    tensorflow.keras.metrics.Recall(name='recall'),
    tensorflow.keras.metrics.AUC(name='AUC')
]

# Training parameters
epochs = 100
batch_size = 256
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adamax(learning_rate = learning_rate)
filepath="model_best_weights.hdf5"    
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')

# Model architecture
inputs = Input(shape=input_shape,name='input')
x = Conv2D(16,(2,2),strides=(1,1),padding='valid',kernel_initializer='normal')(inputs)
x = AveragePooling2D((2,2), strides=(1,1))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
x = Conv2D(32,(2,2), strides=(1, 1), padding="valid",kernel_initializer='normal')(x)
x = AveragePooling2D((2,2), strides=(1,1))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
x = Conv2D(64,(2,2), strides=(1, 1), padding="valid",kernel_initializer='normal')(x)
x = AveragePooling2D((2,2), strides=(1,1))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
x = Conv2D(128,(2,2), strides=(1, 1), padding="valid",kernel_initializer='normal')(x)
x = AveragePooling2D((2,2), strides=(1,1))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
td = Reshape([31,80*128])(x)
x = LSTM(256, return_sequences=True)(td)
x = Activation('tanh')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = AttentionLayer(return_sequences=False)(x)
x = Dense(100)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(1,name='output_layer')(x)
x = Activation('sigmoid')(x)
model = Model(inputs=inputs, outputs=x)

# Compile model
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=METRICS)

# Train model
history = model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint])

# Evaluate model
loss, acc, *_ = model.evaluate(X_valid, y_valid, verbose=0)
covPredict = model.predict(X_valid, verbose=0)
covPredict = np.where(covPredict >= 0.5, 1,0)
real_stat = y_valid

recall = my_recall(real_stat, covPredict)
precision = my_precision(real_stat, covPredict)
roc_auc = my_roc_auc(real_stat, covPredict)
f1_score = my_f1(real_stat, covPredict)
balanced_acc = get_balanced_acc(real_stat, covPredict)
specificity = (2*get_balanced_acc(y_valid,covPredict))-my_recall(y_valid,covPredict)

print('Validation results for final epoch')
print('Accuracy : ', acc)
print('Precision : ', precision)
print('Recall : ', recall)
print('F1 : ', f1_score)
print('ROC AUC : ', roc_auc)
print('Specificity : ', specificity)

custom_objects = {'AttentionLayer': AttentionLayer}
best_model = load_model('model_best_weights.hdf5',custom_objects=custom_objects)
score = best_model.evaluate(X_valid, y_valid, verbose=0)
covPredict = best_model.predict(X_valid, verbose=0)
covPredict = np.where(covPredict > 0.5, 1, 0)

print('Validation results for best model')
print("Accuracy : ",score[1])
print("Precision : ",my_precision(y_valid,covPredict))
print("Recall : ",my_recall(y_valid,covPredict))
print("F1 : ",my_f1(y_valid,covPredict))
print("ROC AUC : ",my_roc_auc(y_valid,covPredict))
print("Specificity : ",(2*get_balanced_acc(y_valid,covPredict))-my_recall(y_valid,covPredict))


## Plot confusion matrix
plotConfusionMatrix(y_valid, covPredict)

histories = [history.history]

## Plot accuracy curves
plotCurves('Attention-based CNN-LSTM train and validation accuracy curves','Accuracy','Epoch','accuracy',histories)

## Plot loss curves
plotCurves('Attention-based CNN-LSTM train and validation loss curves','Loss','Epoch','loss',histories)

## Plot Sensitivity curves
plotCurves('Attention-based CNN-LSTM train and validation sensitivity curves','Sensitivity','Epoch','recall',histories)

## Plot Precision curves
plotCurves('Attention-based CNN-LSTM train and validation precision curves','Precision','Epoch','precision',histories)

## Plot ROC curve

probabilities = best_model.predict(X_valid, verbose=0).ravel()
fpr, tpr, thresholds = roc_curve(y_valid, probabilities, pos_label=1)
auc = auc(fpr, tpr)
plotROCCurve(fpr,tpr,auc,'darkred','Attention-based CNN-LSTM ROC AUC ','Attention-based CNN-LSTM baseline ROC AUC')