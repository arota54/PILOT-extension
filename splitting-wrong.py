from cgi import test
from pyexpat import features
from re import sub
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import collections
import os, shutil
from pathlib import Path

# utilizzo di una GPU su scheda grafica locale
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

base_dir = ''
maldonado_input_file_binary = base_dir + 'features-matrices/maldonado-features-matrices-binary.csv'
maldonado_input_file_multiclass = base_dir + 'features-matrices/maldonado-features-matrices-multiclass.csv'

debthunter_input_file_binary = base_dir + 'features-matrices/debthunter-features-matrices-binary.csv'
debthunter_input_file_multiclass = base_dir + 'features-matrices/debthunter-features-matrices-multiclass.csv'


# for multiclass: 0 - DESIGN, 1 - DEFECT, 2 - IMPLEMENTATION, 3 - DOCUMENTATION, 4 - TEST
# for binary: 0 - non-SATD, 1 - SATD
def labels_from_text_to_int(np_labels_text, is_binary_classification) :
    text = np.array(['DESIGN', 'DEFECT', 'IMPLEMENTATION', 'DOCUMENTATION', 'TEST'])
    labels = []
    
    if is_binary_classification : 
        for x in np_labels_text : 
            if np.where(text == x) == [] :
                labels = np.append(labels, [0])
            else :
                labels = np.append(labels, [1])
    else : 
        for x in np_labels_text :
            labels = np.append(labels, np.where(text == x))
    
    return labels.astype(int)


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


def create_prediction_files(path) :
    df = pd.DataFrame(list())
    df.to_csv(path + 'GroundTruth.csv')
    df.to_csv(path + 'Prediction.csv')


def create_train_and_test(input_file, output_folder, is_binary_classification) :
    delete_folder(output_folder)
    create_prediction_files(output_folder)

    print("\n\nFolder: ", output_folder)

    df = pd.read_csv(input_file)
    print("df shape: ", df.shape)
    
    df_features = df.iloc[:, 1:] # skip only the first column
    
    # convert all the labels from text to int
    labels = df.iloc[:, 0].to_numpy()
    np_labels_text = []
    for l in labels : 
        np_labels_text = np.append(np_labels_text, str(l).split(" - ")[1])
    np_labels = labels_from_text_to_int(np_labels_text, is_binary_classification)

    print("BEFORE: ", df_features.shape, np_labels.shape)
    print(collections.Counter(np_labels))

    df_dataset = df_features
    # create a new column to insert the label number
    df_dataset.insert(df_dataset.shape[1], 'n_label', np_labels.tolist()) 
    # create a new column to insert the number of split set in order to then create 10 different sets
    np_zeros = np.zeros(shape=(df_features.shape[0],)).astype(int)
    df_dataset.insert(df_dataset.shape[1], 'n_split_set', np.zeros(df_features.shape[0], dtype=np.int64)) 
    
    # shuffle the dataset
    df_dataset = df_dataset.sample(frac=1, random_state=12)#.reset_index(drop=True)
    print(df_dataset.head(), df_dataset.shape)

    if is_binary_classification :
        df_dataset = split_in_ten_sets_binary(df_dataset)
    else :
        df_dataset = split_in_ten_sets_multiclass(df_dataset)
    
    sum = 0
    # save dataframes into 10 folders (Round1, ...)
    for set_number in range(1, 11) :
        print("\n SET ", set_number)
        subset = df_dataset.loc[df_dataset['n_split_set'] == set_number]
        #print(subset.head())
        sum = sum + subset.shape[0]
        #print("SET", set_number, subset.shape)
        #print(subset['n_label'].value_counts())

        whole_set = subset.iloc[:, :subset.shape[1]-1]#.reset_index()
        features_set = whole_set.loc[:, whole_set.columns != 'n_label']#.reset_index()
        labels_set = whole_set['n_label']
        #print(labels_set.value_counts())
        assert features_set.shape[0] == labels_set.shape[0]
        assert (np.array(features_set.index) == np.array(labels_set.index)).all()
        """print(whole_set.head())
        print(features_set.head())
        print(labels_set.head())
        print(features_set.index)
        print(labels_set.index)"""

        x, y = np.array(features_set), np.array(labels_set)

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        for train_index, test_index in splitter.split(x, y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = x[train_index], y[test_index]
            y_train, y_test = y[train_index], y[test_index]


        #x_train, x_test, y_train, y_test = train_test_split(features_set, labels_set, test_size=0.1, random_state=42, stratify=labels_set)
        print("TRAIN: ", x_train.shape, y_train.shape)
        #print(np.unique(y_train))
        unique_elements, counts_elements = np.unique(y_train, return_counts=True)
        print(np.asarray((unique_elements, counts_elements)))
        print("TEST: ", x_test.shape, y_test.shape)
        #print(np.unique(y_test))
        unique_elements, counts_elements = np.unique(y_test, return_counts=True)
        print(np.asarray((unique_elements, counts_elements)))

        
        # check dimension of the split and every label in every set
        assert x_train.shape[0] + x_test.shape[0] == whole_set.shape[0]
        if is_binary_classification :
            val = (0 in y_train and 1)
            assert val == True
            val = (0 in y_test and 1)
            assert val == True
        else :
            val = (0 in y_train and 1 in y_train and 2 in y_train and 3 in y_train and 4 in y_train)
            assert val == True
            val = (0 in y_test and 1 in y_test and 2 in y_test and 3 in y_test and 4 in y_test)
            assert val == True

        #print(x_train)

        path = output_folder + 'Round' + str(set_number)
        os.mkdir(path)            
        # save train and test as csv
        np.savetxt(path + '/training_data.csv', x_train, delimiter=",", fmt="%f")
        np.savetxt(path + '/training_labels.csv', y_train.astype(int), delimiter=",", fmt="%i")
        np.savetxt(path + '/testing_data.csv', x_test, delimiter=",", fmt="%f")
        np.savetxt(path + '/testing_labels.csv', y_test.astype(int), delimiter=",", fmt="%i")

        
    

    # check that there are no leftovers
    assert df_dataset.shape[0] == sum
    assert df_dataset.loc[df_dataset['n_split_set'] == 0].empty == True

    


# assign to the index in the split their own set_number
def assign_set_number(df_dataset, split) :
    for set_number in range(1, 11) :
            sub_split = split[set_number-1]

            for index in sub_split : 
                df_dataset.at[index, 'n_split_set'] = set_number
                #print(df_dataset.at[index, 'n_split_set'])
        
    return df_dataset


def split_in_ten_sets_binary(df_dataset) :
    # 0-1 (labels)
    for label in range(2) :
        # take the subset with that specific label number
        df_based_on_label = df_dataset.loc[df_dataset['n_label'] == label]

        # take and split indexes into 10 
        indexes = df_based_on_label.index.to_numpy()
        split = np.array_split(indexes, 10)

        # check the split operation
        assert df_based_on_label.shape[0] == sum([sub.size for sub in indexes])

        df_dataset = assign_set_number(df_dataset, split)

        # check there are no leftovers (rows with n_split_set to be assigned)
        assert df_dataset.loc[(df_dataset['n_split_set'] == 0) & (df_dataset['n_label'] == label)].empty == True

    return df_dataset


def split_in_ten_sets_multiclass(df_dataset) :
    # from 0 to 4 (labels)
    for label in range(5) :
        # take the subset with that specific label number
        df_based_on_label = df_dataset.loc[df_dataset['n_label'] == label]

        # take and split indexes into 10 
        indexes = df_based_on_label.index.to_numpy()
        split = np.array_split(indexes, 10)

        # check the split operation
        assert df_based_on_label.shape[0] == sum([sub.size for sub in indexes])

        df_dataset = assign_set_number(df_dataset, split)
            
        # check there are no leftovers (rows with n_split_set to be assigned)
        assert df_dataset.loc[(df_dataset['n_split_set'] == 0) & (df_dataset['n_label'] == label)].empty == True
        
    return df_dataset
    


create_train_and_test(debthunter_input_file_multiclass, "DatasetD2/multiclass/", False)

create_train_and_test(maldonado_input_file_multiclass, "DatasetD1/multiclass/", False)