import csv
import string
import pandas as pd
import tensorflow as tf
import numpy as np
import os, shutil
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
import collections

# utilizzo di una GPU su scheda grafica locale
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

base_dir = ''

maldonado_features_matrices_binary = base_dir + 'tf-idf/features-matrices/features-matrices-maldonado-binary.csv'
maldonado_features_matrices_multiclass = base_dir + 'tf-idf/features-matrices/features-matrices-maldonado-multiclass.csv'
maldonado_output_folder_multiclass = base_dir + 'tf-idf/multiclass/DatasetD1/'
maldonado_output_folder_binary = base_dir + 'tf-idf/binary/DatasetD1/'

debthunter_features_matrices_binary = base_dir + 'tf-idf/features-matrices/features-matrices-debthunter-binary.csv'
debthunter_features_matrices_multiclass = base_dir + 'tf-idf/features-matrices/features-matrices-debthunter-multiclass.csv'
debthunter_output_folder_multiclass = base_dir + 'tf-idf/multiclass/DatasetD2/'
debthunter_output_folder_binary = base_dir + 'tf-idf/binary/DatasetD2/'

def delete_folder(path) :
    # delete folder and everything within, if it exists
    if(os.path.isdir(path)) :
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        #os.rmdir(path)


# for multiclass: 0 - DESIGN, 1 - DEFECT, 2 - IMPLEMENTATION, 3 - DOCUMENTATION, 4 - TEST
# for binary: 0 - non-SATD, 1 - SATD
def labels_from_text_to_int(np_labels_text) :
    text = np.array(['DESIGN', 'DEFECT', 'IMPLEMENTATION', 'DOCUMENTATION', 'TEST'])
    labels = []
    
    if BINARY_CLASSIFICATION : 
        for x in np_labels_text : 
            if(x == 'WITHOUT_CLASSIFICATION') :
                labels = np.append(labels, [0])
            else :
                labels = np.append(labels, [1])
    else : 
        for x in np_labels_text :
            labels = np.append(labels, np.where(text == x))
    
    return labels.astype(int)


def main(binary_classification, dataset) :
    global BINARY_CLASSIFICATION, DATASET, INPUT_FILE, OUTPUT_FOLDER

    BINARY_CLASSIFICATION = binary_classification
    DATASET = dataset

    if DATASET:
        if BINARY_CLASSIFICATION:
            INPUT_FILE = maldonado_features_matrices_binary
            OUTPUT_FOLDER = maldonado_output_folder_binary #'tf-idf/binary/DatasetD1/'
        else:
            INPUT_FILE = maldonado_features_matrices_multiclass
            OUTPUT_FOLDER = maldonado_output_folder_multiclass #'tf-idf/multiclass/DatasetD1/'
    else:
        if BINARY_CLASSIFICATION:
            INPUT_FILE = debthunter_features_matrices_binary
            OUTPUT_FOLDER = debthunter_output_folder_binary #'tf-idf/binary/DatasetD2/'
        else:
            INPUT_FILE = debthunter_features_matrices_multiclass
            OUTPUT_FOLDER = debthunter_output_folder_multiclass #'tf-idf/multiclass/DatasetD2/'

    delete_folder(OUTPUT_FOLDER)

    print("\n\nFolder: ", OUTPUT_FOLDER)
    print("Read features matrices csv file")

    df = pd.read_csv(INPUT_FILE)
    print("df shape: ", df.shape)
    
    df_features = df.iloc[:, 1:] # skip only the first column
    
    # convert all the labels from text to int
    labels = df.iloc[:, 0].to_numpy()
    np_labels_text = []
    for l in labels : 
        np_labels_text = np.append(np_labels_text, str(l).split(" - ")[1])
    np_labels = labels_from_text_to_int(np_labels_text)

    print("BEFORE: ", df_features.shape, np_labels.shape)
    print(collections.Counter(np_labels))

    df_dataset = df_features
    # create a new column to insert the label number
    df_dataset.insert(df_dataset.shape[1], 'n_label', np_labels.tolist()) 

    # shuffle the dataset
    df_dataset = df_dataset.sample(frac=1, random_state=12)#.reset_index(drop=True)
    print(df_dataset.head(), df_dataset.shape)


    df_features = df_dataset.loc[:, df_dataset.columns != 'n_label']#.reset_index()
    df_labels = df_dataset['n_label']
    print(df_features.shape, df_labels.shape)


    # save dataframes into 10 folders (Round1, ...)
    set_number = 1
    kf = KFold(n_splits=10, shuffle=True, random_state=42, )
    print(kf)

    x, y = np.array(df_features), np.array(df_labels)


    for train_val_index, test_index in kf.split(df_features, df_labels):
        print("\nSET: ", set_number)

        x_train_val, y_train_val = df_features.iloc[train_val_index], df_labels.iloc[train_val_index] 
        x_test, y_test = df_features.iloc[test_index], df_labels.iloc[test_index] 

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
        for train_index, validation_index in splitter.split(x_train_val, y_train_val):
            x_validation, y_validation = x_train_val.iloc[validation_index], y_train_val.iloc[validation_index]
            x_train, y_train = x_train_val.iloc[train_index], y_train_val.iloc[train_index]
    
        #print(x_train)
        print("TRAIN: ", x_train.shape, y_train.shape)
        unique_elements, counts_elements = np.unique(y_train, return_counts=True)
        print(np.asarray((unique_elements, counts_elements)))
        print("VALIDATION: ", x_validation.shape, y_validation.shape)
        unique_elements, counts_elements = np.unique(y_validation, return_counts=True)
        print(np.asarray((unique_elements, counts_elements)))
        print("TEST: ", x_test.shape, y_test.shape)
        unique_elements, counts_elements = np.unique(y_test, return_counts=True)
        print(np.asarray((unique_elements, counts_elements)))

        

        # check dimension of the split and every label in every set
        assert x_train.shape[0] + x_validation.shape[0] + x_test.shape[0] == df_features.shape[0]
        if BINARY_CLASSIFICATION :
            val = (0 in y_train.to_numpy() and 1 in y_train.to_numpy())
            assert val == True
            val = (0 in y_validation.to_numpy() and 1 in y_validation.to_numpy())
            assert val == True
            val = (0 in y_test.to_numpy() and 1 in y_test.to_numpy())
            assert val == True
        else :
            val = (0 in y_train and 1 in y_train and 2 in y_train and 3 in y_train and 4 in y_train)
            assert val == True
            val = (0 in y_test and 1 in y_test and 2 in y_test and 3 in y_test and 4 in y_test)
            assert val == True
        
        
        path = OUTPUT_FOLDER + 'Round' + str(set_number)
        os.mkdir(path)   

        # save train, validation and test as csv
        np.savetxt(path + '/training_data.csv', x_train, delimiter=",", fmt="%s")
        np.savetxt(path + '/training_labels.csv', y_train.astype(int), delimiter=",", fmt="%i")
        np.savetxt(path + '/validation_data.csv', x_validation, delimiter=",", fmt="%s")
        np.savetxt(path + '/validation_labels.csv', y_validation.astype(int), delimiter=",", fmt="%i")
        np.savetxt(path + '/testing_data.csv', x_test, delimiter=",", fmt="%s")
        np.savetxt(path + '/testing_labels.csv', y_test.astype(int), delimiter=",", fmt="%i")

        set_number = set_number + 1 


# BINARY_CLASSIFICATION = True prende l'intero dataset (WITHOUT_CLASSIFICATION + tutte le altre etichette)
# BINARY_CLASSIFICATION = False prende solo le altre etichette (non prende WITHOUT_CLASSIFICATION)
# DATASET = True (Maldonado), DATASET = False (DebtHunter)
#main(True, False) # DebtHunter binary
main(True, True) # Maldonado binary