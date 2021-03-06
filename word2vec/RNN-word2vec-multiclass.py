import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from numpy import savetxt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
from sklearn.model_selection import StratifiedShuffleSplit
from keras_tqdm import TQDMNotebookCallback
from keras import regularizers
from torch import dropout
import os
import time
from tensorflow.keras import models, layers, optimizers
from keras.utils.vis_utils import plot_model


from keras.layers import Dense, Input, GlobalMaxPooling1D, MaxPool1D, GlobalMaxPool1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dropout, SpatialDropout1D
from keras.models import Model

"""os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.compat.v1.Session(config=config) """

start_time = time.time()

base_dir = ''
epochs = 5
batch_size = 16

# utilizzo di una GPU su scheda grafica locale
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def plot_history(loss, accuracy, val_loss, val_accuracy, path) :
    x_plot = list(range(1, epochs*10 + 1))

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, loss)
    plt.plot(x_plot, val_loss)
    plt.legend(['Training', 'Validation'])

    plt.savefig(path + 'RNN-loss.png')

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, accuracy)
    plt.plot(x_plot, val_accuracy)
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig(path + 'RNN-accuracy.png')
    plt.show()

    


def predict_labels(x_test, model) :
    predictions = model.predict(x_test)
    predictions_labels = np.argmax(predictions, axis=1)
    """predictions_labels = (predictions > 0.5).astype(np.int8)
    print(np.equal(predictions_labels, np.round(predictions)).all())"""
    return predictions_labels


def plot_cm(predictions, y_test, path) :
    print(classification_report(y_test, predictions))
    
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10,10), dpi = 70)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='YlGnBu')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    plt.savefig(path + 'RNN-cm.png')


# load training, validation and testing labels from CSV files
def load_labels(path) :
    """==============read training labels=============="""          
    raw_data = open(path+'/training_labels.csv', 'rt')
    tr_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Train: ", collections.Counter(tr_l))

    """==============read validation data=============="""
    raw_data = open(path+'/validation_labels.csv', 'rt')
    val_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Validation: ", collections.Counter(val_l))
    
    """==============read testing data=============="""
    raw_data = open(path+'/testing_labels.csv', 'rt')
    te_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Test: ", collections.Counter(te_l))
    
    return (tr_l, val_l, te_l)


# load training, validation and testing sequences and emb_matrix (word2vec) from CSV files
def load_sequences_and_matrix(path) :
    """==============read training sequences=============="""
    raw_data = open(path+'/train_sequences.csv', 'rt')
    tr_s = np.loadtxt(raw_data, delimiter=",")
    print("Train sequences: ", tr_s.shape)

    """==============read validation sequences=============="""
    raw_data = open(path+'/validation_sequences.csv', 'rt')
    val_s = np.loadtxt(raw_data, delimiter=",")
    print("Validation sequences: ", val_s.shape)
    
    """==============read testing sequences=============="""
    raw_data = open(path+'/test_sequences.csv', 'rt')
    te_s = np.loadtxt(raw_data, delimiter=",")
    print("Test sequences: ", te_s.shape)

    """==============read emb_matrix=============="""
    raw_data = open(path+'/emb_matrix.csv', 'rt')
    emb_matrix = np.loadtxt(raw_data, delimiter=",")
    print("Emb_matrix: ", emb_matrix.shape)
    
    return (tr_s, val_s, te_s, emb_matrix)


def create_model(train_sequences, emb_matrix) : 
    print(emb_matrix.shape[1])

    # result-1: 8 epochs, 128 batch_size
    input_ = layers.Input(shape = train_sequences[0,:].shape, )
    x = layers.Embedding(7000+1, emb_matrix.shape[1], weights=[emb_matrix], trainable=False)(input_)
    #x = layers.Bidirectional(layers.LSTM(64))(x) # LSTM layer
    x = layers.LSTM(64, dropout=0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    #x = layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = models.Model(input_, output)
    opt = optimizers.Adam(learning_rate=0.01, beta_1=0.9)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # result-2: 5 epochs, 64 batch_size
    """input_ = layers.Input(shape = train_sequences[0,:].shape, )
    x = layers.Embedding(5097+1, emb_matrix.shape[1], weights=[emb_matrix], trainable=False)(input_)
    #x = layers.Bidirectional(layers.LSTM(64))(x) # LSTM layer
    x = layers.LSTM(64, dropout=0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = models.Model(input_, output)
    opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])"""
    
    model.summary()

    plot_model(model, to_file=base_dir + 'word2vec/RNN-results/RNN-model.png', show_shapes=True, show_layer_names=True)
    
    return model


def model_fit(train_sequences, train_labels, validation_sequences, validation_labels, model) :    
    """print("train data: ", training_data.shape)
    print("train labels: ", training_labels.shape)"""

    history = model.fit(train_sequences, train_labels, 
                        epochs = epochs, 
                        validation_data = (validation_sequences, validation_labels), 
                        batch_size = batch_size,
                        verbose=2)
    return history


def run(path) :
    print("Load labels")
    y_train, y_val, y_test = load_labels(path)
    print("Load sequences")
    train_sequences, validation_sequences, test_sequences, emb_matrix = load_sequences_and_matrix(path)

    history = []
    predictions = []
    model = create_model(train_sequences, emb_matrix) #number of features
    history = model_fit(train_sequences, y_train, validation_sequences, y_val, model)
    predictions = predict_labels(test_sequences, model)
    return (history, predictions, y_test)


def runExperiment(root):
    loss = []
    accuracy = []
    val_loss = []
    val_accuracy = [] 
    predictions = []
    y_test = []
    
    for i in range(10):
        path = root+'Round'+str(i+1)
        print(path)
        
        history, round_predictions, round_y_test = run(path)
        
        print(history.history['loss'])
        
        loss = np.append(loss, history.history['loss'])
        accuracy = np.append(accuracy, history.history['accuracy'])
        val_loss = np.append(val_loss, history.history['val_loss'])
        val_accuracy = np.append(val_accuracy, history.history['val_accuracy'])
        predictions = np.append(predictions, round_predictions)
        y_test = np.append(y_test, round_y_test)
        
        print("Seconds: ", (time.time() - start_time))
        print("\n")
        
    return (loss, accuracy, val_loss, val_accuracy, predictions, y_test)


#loss, accuracy, val_loss, val_accuracy, predictions, y_test = runExperiment(base_dir + "word2vec/multiclass/DatasetD1/")
loss, accuracy, val_loss, val_accuracy, predictions, y_test = runExperiment(base_dir + "word2vec/multiclass/pilot/")
#loss, accuracy, val_loss, val_accuracy, predictions, y_test = runExperiment(base_dir + "word2vec/binary/DatasetZhao/")
#loss, accuracy, val_loss, val_accuracy, predictions, y_test = runExperiment(base_dir + "smote/word2vec/multiclass/DatasetD1/")

print(loss.shape)
print(accuracy.shape)
print(val_loss.shape)
print(val_accuracy.shape)
print(predictions.shape)
print(y_test.shape)


plot_cm(predictions, y_test, base_dir + "word2vec/multiclass/RNN-results/")
plot_history(loss, accuracy, val_loss, val_accuracy, base_dir + "word2vec/multiclass/RNN-results/")



np.savetxt(base_dir + "word2vec/multiclass/RNN-results/RNN-Prediction.csv", predictions.T.astype(int), delimiter=",", fmt="%i")
np.savetxt(base_dir + "word2vec/multiclass/RNN-results/RNN-Truth.csv", y_test.T.astype(int), delimiter=",", fmt="%i")
np.savetxt(base_dir + "word2vec/multiclass/RNN-results/RNN-Loss.csv", loss.T.astype(float), delimiter=",", fmt="%f")
np.savetxt(base_dir + "word2vec/multiclass/RNN-results/RNN-Val_Loss.csv", val_loss.T.astype(float), delimiter=",", fmt="%f")
np.savetxt(base_dir + "word2vec/multiclass/RNN-results/RNN-Acc.csv", accuracy.T.astype(float), delimiter=",", fmt="%f")
np.savetxt(base_dir + "word2vec/multiclass/RNN-results/RNN-Val_Acc.csv", val_accuracy.T.astype(float), delimiter=",", fmt="%f")

print("Total time: ", (time.time() - start_time))

