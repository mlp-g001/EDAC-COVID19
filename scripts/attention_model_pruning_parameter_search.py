import os
import time
import numpy as np
import pandas as pd
import zipfile
import tempfile
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_model_optimization as tfmot
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

from sklearn.metrics import roc_auc_score

from attention_layer import AttentionLayer

tf.random.set_seed(1234)

def load_set(data_dir, set_name):
    '''
        Function to load a dataset based on the set name.
    '''

    path = os.path.join(data_dir, f'{set_name}_coughvid_melspec.npz')
    features = np.load(path)
    X = features['images']
    y = features['covid_status']
    
    return X, y

class PrunableAttentionLayer(AttentionLayer, tfmot.sparsity.keras.PrunableLayer):
    '''
        Prunable implementation of AttentionLayer

        The Attention Layer is not pruned as it is recommended to avoid pruning of
        attention mechanisms.
    '''
    def get_prunable_weights(self):
        return []

def setup_model(input_shape=(39, 88, 3)):
    '''
        Function to setup model for pruning experiments.
    '''
    
    inputs = Input(shape=input_shape, name='input', dtype='float32')
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
    x = PrunableAttentionLayer(return_sequences=False)(x)
    x = Dense(100)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, name='output_layer')(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    
    return model

def predict(model, X, quantise=False, float16=False, batch_size=119, N=100):
    '''
        Function to evaluate models in terms of qzipped file size and inference time.
        Predicted labels are also return for further evaulation.

        Label prediction is done in batches specified by the batch_size parameters, while
        average inference time is calculated for the first N individual predicitons.
    '''

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    
    if quantise:
        # 8-bit quantisation
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if float16:  
        def dataset_gen():
            '''
                Representative dataset generator.
            '''
            for x in X:
                yield np.reshape(x, (1,) + x.shape)
        
        # 16-bit float quantisation
        converter.target_spec.supported_types = [tf.float16]
        converter.representative_dataset = dataset_gen
        
    quantized_tflite_model = converter.convert()
    
    # Check size difference
    _, tmp_file = tempfile.mkstemp('.tflite')
    with open(tmp_file, 'wb') as f:
        f.write(quantized_tflite_model)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(tmp_file)
    
    size = os.path.getsize(zipped_file) / 10**6
    
    # Remove temp files
    os.remove(tmp_file)
    os.remove(zipped_file)
    
    interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
    
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    # Run N individual inferences for average inference time calculation
    y_pred = []
    start_time = time.time()
    for i in range(N):
        input_data = X[i].reshape(input_shape).astype(input_dtype)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = (interpreter.get_tensor(output_details[0]['index'])[0] > 0.5).astype(int)
        y_pred.append(prediction)
    
    inference_time = (time.time() - start_time) / N
    
    # Change input shape to accept batches
    input_details = interpreter.get_input_details()
    batch_input_shape = (batch_size,) + X.shape[1:]
    
    interpreter.resize_tensor_input(input_details[0]['index'], batch_input_shape)
    
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    # Batched input
    X_batched = X.reshape((-1, ) + batch_input_shape)

    y_pred = []
    for i in range(X_batched.shape[0]):
        input_data = X_batched[i].reshape(input_shape).astype(input_dtype)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = (interpreter.get_tensor(output_details[0]['index']) > 0.5).astype(int)
        y_pred.append(prediction)
    
    return np.array(y_pred).flatten(), size, inference_time

def evaluate(model, X, y, batch_size=119, N=100):
    '''
        Function to evaulate 3 configuration for experiments.
    '''
    results = dict()
    
    # Default evaulation
    y_pred, size, inf_time = predict(model, X, quantise=False, float16=False, batch_size=batch_size, N=N)
    auc = roc_auc_score(y, y_pred)
    
    results['default'] = {
        'size': size,
        'inf_time': inf_time,
        'auc': auc
    }
    
    # Int8 post training quantisation
    y_pred, size, inf_time = predict(model, X, quantise=True, float16=False, batch_size=batch_size, N=N)
    auc = roc_auc_score(y, y_pred)
    
    results['int8'] = {
        'size': size,
        'inf_time': inf_time,
        'auc': auc
    }
    
    # Float16 post training quantisation
    y_pred, size, inf_time = predict(model, X, quantise=True, float16=True, batch_size=batch_size, N=N)
    auc = roc_auc_score(y, y_pred)
    
    results['float16'] = {
        'size': size,
        'inf_time': inf_time,
        'auc': auc
    }
    
    return results

def run_exp(model_weights_path,
            optimizer,
            loss,
            metrics,
            X_train, y_train,
            X_valid, y_valid,
            pruning_percentages,
            experiment='const',
            batch_size=32,
            epochs=10,
            verbose=0):
    
    '''
        Function to run pruning schedule experiments. Constant Sparsity and
        Polynomial Decay schedules can be selected by setting the experiment
        parameter as 'const' and 'poly' respectively.
    
    '''
    
    experiment_names = {'const': 'Constant Sparsity',
                        'poly':  'Polynomial Decay'}
    
    curr_experiment = experiment_names[experiment]    
    
    # Dictionary to store results
    exp_results = dict()
    
    # Instantiate pretrained model
    base_model = setup_model()
    base_model.load_weights(model_weights_path)
    base_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # Calculate metrics for base model and map to 0% pruning sparsity
    exp_results['0.0'] = evaluate(base_model, X_valid, y_valid)
    
    for percent in pruning_percentages:
        print(f"Running {curr_experiment} experiment with {percent*100:.2f}% pruning.")
        
        # Reset weights to base model
        base_model.load_weights(model_weights_path)

        if experiment == 'const':
            pruning_params = {'pruning_schedule':
                              tfmot.sparsity.keras.ConstantSparsity(percent, 
                                                                    0, 
                                                                    end_step=-1, 
                                                                    frequency=1)}
        elif experiment == 'poly':
            pruning_params = {'pruning_schedule':
                              tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0,
                                                                   final_sparsity=percent,
                                                                   begin_step=0,
                                                                   end_step=np.ceil(X_train.shape[0] / batch_size).astype(np.int32) * epochs)}
        
        # Creating pruning model
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
        model_for_pruning.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

        # Fine-tune pruned model
        model_for_pruning.fit(X_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(X_valid, y_valid),
                              callbacks=callbacks,
                              verbose=verbose)

        # Apply mask
        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

        # Calculate metrics and map to pruning sparsity
        exp_results[str(percent)] = evaluate(model_for_export, X_valid, y_valid)
    
    return exp_results

def save_results(exp_results, name):
    '''
        Function to save results of experiment.
    '''
    with open(f'{name}_results.json', 'w') as f:
        json.dump(exp_results, f)

def main():
    ####################################################################
    # Load datasets
    data_dir = '/kaggle/input/coughvid-melspectrum-extracted/coughvid_melspec'
    X_train, y_train = load_set(data_dir, 'train')    
    X_valid, y_valid = load_set(data_dir, 'valid')
    
    ####################################################################
    # Load model weights
    weights_dir = '/kaggle/input/attention-model-weights'
    model_name = 'model_best_weights.hdf5'
    model_path = os.path.join(weights_dir, model_name)

    ####################################################################
    # Evaluation metrics
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='AUC')
    ]
    
    ####################################################################
    # Training parameters
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    loss = 'binary_crossentropy'
    epochs = 10
    
    ####################################################################
    # Experiment parameters
    pruning_percentages = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
    experiment = 'const'
    experiment_name = 'attention_const_sparsity'

    ####################################################################
    # Run experiment
    exp_results = run_exp(model_path,
                          optimizer,
                          loss,
                          METRICS,
                          X_train, y_train,
                          X_valid, y_valid,
                          pruning_percentages,
                          experiment=experiment,
                          epochs=epochs)

    # Save results of experiment
    save_results(exp_results, experiment_name)

if __name__ == '__main__':
    main()