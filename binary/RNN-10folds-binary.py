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
from keras_tqdm import TQDMNotebookCallback


base_dir = ''
epochs = 100
batch_size = 32

# utilizzo di una GPU su scheda grafica locale
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def plot_history(loss, accuracy, val_loss, val_accuracy, path) :
    x_plot = list(range(1, epochs + 1))

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
    predictions_labels = np.argmax(predictions, axis=1)
    return predictions_labels


def plot_cm(predictions, y_test) :
    print(classification_report(y_test, predictions))
    
    cm = confusion_matrix(y_test, predictions)
    plt.figure()
    plt.figure(figsize=(10,10), dpi = 70)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='YlGnBu')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')


# load training and testing data from CSV files
def load_data(path) :
    """==============read training data=============="""
    raw_data = open(path+'/training_data.csv', 'rt')
    tr_d = np.loadtxt(raw_data, delimiter=",") # np array
    tr_d = np.resize(tr_d, (tr_d.shape[0], 1, tr_d.shape[1]))
          
    raw_data = open(path+'/training_labels.csv', 'rt')
    tr_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Train: ", tr_d.shape, collections.Counter(tr_l))
    
    """==============read testing data=============="""
    raw_data = open(path+'/testing_data.csv', 'rt')
    te_d = np.loadtxt(raw_data, delimiter=",")
    print("Test: ", te_d.shape)
    te_d = np.resize(te_d, (te_d.shape[0], 1, te_d.shape[1]))

    raw_data = open(path+'/testing_labels.csv', 'rt')
    te_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Test: ", te_d.shape, collections.Counter(te_l))
    
    return (tr_d, tr_l, te_d, te_l)


def create_model(shape) :
    model = tf.keras.Sequential()
    x = tf.keras.layers.Input(shape=(1, shape))
    #xx = tf.keras.layers.Reshape((1372, 1), input_shape=(1372, ))(x)
    #xx = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, recurrent_regularizer='l2'))(x)
    xx = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    """xx = tf.keras.layers.Dense(512, activation='relu')(xx)
    xx = tf.keras.layers.Dense(256, activation='relu')(xx)
    xx = tf.keras.layers.Dense(128, activation='relu')(xx)"""
    xx = tf.keras.layers.Dense(64, activation='relu')(xx)
    xx = tf.keras.layers.Dense(5, activation='softmax')(xx)
    model = tf.keras.Model(x, xx)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  metrics=['accuracy'])
    
    model.summary()
    
    return model


def model_fit(training_data, training_labels, model) :    
    print("train data: ", training_data.shape)
    print("train labels: ", training_labels.shape)


    history = model.fit(training_data, training_labels, 
                        epochs = epochs, 
                        validation_split = 0.1, 
                        batch_size = batch_size,
                        verbose=2)
    return history


def run(path) :
    x_train, y_train, x_test, y_test = load_data(path)
    history = []
    predictions = []
    model = create_model(x_train.shape[2]) #number of features
    history = model_fit(x_train, y_train, model)
    predictions = predict_labels(x_test, model)
    return (history, predictions, y_test)


def runExperiment(root):
    loss = []
    accuracy = []
    val_loss = []
    val_accuracy = [] 
    predictions = []
    y_test = []
    
    #for i in range(1):
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
        
        print("\n")
        
    return (loss, accuracy, val_loss, val_accuracy, predictions, y_test)



loss, accuracy, val_loss, val_accuracy, predictions, y_test = runExperiment(base_dir + "binary/DatasetD2/rounds/")

print(loss.shape)
print(accuracy.shape)
print(val_loss.shape)
print(val_accuracy.shape)
print(predictions.shape)
print(y_test.shape)

plot_history(loss, accuracy, val_loss, val_accuracy, base_dir + "binary/DatasetD2/rounds/")

plot_cm(predictions, y_test)

#predictions.T.to_csv(base_dir + "")