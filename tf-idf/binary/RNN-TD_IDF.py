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
from tensorflow.keras import models, layers, optimizers


from keras.layers import Dense, Input, GlobalMaxPooling1D, MaxPool1D, GlobalMaxPool1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dropout
from keras.models import Model

"""os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.compat.v1.Session(config=config) """

base_dir = ''
epochs = 5
batch_size = 64

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

    plt.savefig(path + 'loss.png')

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, accuracy)
    plt.plot(x_plot, val_accuracy)
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig(path + 'accuracy.png')
    plt.show()

    


def predict_labels(x_test, model) :
    predictions = model.predict(x_test)
    #predictions_labels = np.argmax(predictions, axis=1)
    predictions_labels = (predictions > 0.5).astype(np.int8)
    print(np.equal(predictions_labels, np.round(predictions)).all())
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

    plt.savefig(path + 'cm.png')


# load training and testing data from CSV files
def load_data(path) :
    """==============read training data=============="""
    raw_data = open(path+'/training_data.csv', 'rt')
    tr_d = np.loadtxt(raw_data, delimiter=",") # np array
    #tr_d = np.resize(tr_d, (tr_d.shape[0], 1, tr_d.shape[1]))
          
    raw_data = open(path+'/training_labels.csv', 'rt')
    tr_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Train: ", tr_d.shape, collections.Counter(tr_l))

    """==============read validation data=============="""
    raw_data = open(path+'/validation_data.csv', 'rt')
    val_d = np.loadtxt(raw_data, delimiter=",")
    #val_d = np.resize(val_d, (tr_d.shape[0], 1, val_d.shape[1]))

    raw_data = open(path+'/validation_labels.csv', 'rt')
    val_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Validation: ", val_d.shape, collections.Counter(val_l))
    
    """==============read testing data=============="""
    raw_data = open(path+'/testing_data.csv', 'rt')
    te_d = np.loadtxt(raw_data, delimiter=",")
    #te_d = np.resize(te_d, (te_d.shape[0], 1, te_d.shape[1]))

    raw_data = open(path+'/testing_labels.csv', 'rt')
    te_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Test: ", te_d.shape, collections.Counter(te_l))
    
    return (tr_d, tr_l, val_d, val_l, te_d, te_l)


def create_model(shape) : 
    print("Shape features: ", shape)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=shape, output_dim=64, trainable=False),
            tf.keras.layers.LSTM(64, dropout=0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    )
    
    #opt = optimizers.Adam(learning_rate=1, beta_1=0.9)
    opt = optimizers.SGD(learning_rate=0.05)
    model.compile(optimizer=opt, 
                  loss=tf.keras.losses.BinaryCrossentropy(), 
                  metrics=['accuracy'])
    
    model.summary()
    
    return model


def model_fit(x_train, y_train, x_val, y_val, model) :    
    """print("train data: ", training_data.shape)
    print("train labels: ", training_labels.shape)"""

    history = model.fit(x_train, y_train, 
                        epochs = epochs, 
                        validation_data = (x_val, y_val), 
                        batch_size = batch_size,
                        verbose=2)
    return history


def run(path) :
    print("Load data")
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(path)

    print(x_train[0])

    history = []
    predictions = []
    model = create_model(x_train.shape[1]) #number of features
    history = model_fit(x_train, y_train, x_val, y_val, model)
    predictions = predict_labels(x_test, model)
    return (history, predictions, y_test)


def runExperiment(root):
    loss = []
    accuracy = []
    val_loss = []
    val_accuracy = [] 
    predictions = []
    y_test = []
    
    for i in range(10):
    #for i in range(10):
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
        
        print("\n")
        
    return (loss, accuracy, val_loss, val_accuracy, predictions, y_test)



loss, accuracy, val_loss, val_accuracy, predictions, y_test = runExperiment(base_dir + "tf-idf/binary/DatasetD1/")

print(loss.shape)
print(accuracy.shape)
print(val_loss.shape)
print(val_accuracy.shape)
print(predictions.shape)
print(y_test.shape)


plot_cm(predictions, y_test, base_dir + "tf-idf/binary/DatasetD1/")
plot_history(loss, accuracy, val_loss, val_accuracy, base_dir + "tf-idf/binary/DatasetD1/")



np.savetxt(base_dir + "tf-idf/binary/DatasetD1/Prediction.csv", predictions.T.astype(int), delimiter=",", fmt="%i")
np.savetxt(base_dir + "tf-idf/binary/DatasetD1/Truth.csv", y_test.T.astype(int), delimiter=",", fmt="%i")

