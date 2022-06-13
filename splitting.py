from cgi import test
from pyexpat import features
from re import sub
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import collections
import os

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


def create_train_and_test(input_file, output_folder, is_binary_classification) :
    df = pd.read_csv(input_file)
    print("df shape: ", df.shape)
    
    df_features = df.iloc[:, 1:]
    
    # convert all the labels from text to int
    labels = df.iloc[:, 0].to_numpy()
    np_labels_text = []
    for x in labels : 
        np_labels_text = np.append(np_labels_text, str(x).split(" - ")[1])
    np_labels = labels_from_text_to_int(np_labels_text, is_binary_classification)

    print("BEFORE: ", df_features.shape, np_labels.shape)
    print(collections.Counter(np_labels))

    df_dataset = df_features
    # create a new column to insert the label number
    df_dataset.insert(df_dataset.shape[1], 'n_label', np_labels.tolist()) 
    # create a new column to insert the number of split set in order to then create 10 different sets
    np_zeros = np.zeros(shape=(df_features.shape[0],)).astype(int)
    df_dataset.insert(df_dataset.shape[1], 'n_split_set', np.zeros(df_features.shape[0], dtype=np.int64)) 
    
    # shuffle the dataset and reset the index
    df_dataset = df_dataset.sample(frac=1, random_state=12).reset_index(drop=True)
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
        sum = sum + subset.shape[0]
        #print("SET", set_number, subset.shape)
        #print(subset['n_label'].value_counts())

        features_set = subset.iloc[:, :subset.shape[1]-2].reset_index()
        labels_set = subset['n_label']
        assert features_set.shape[0] == labels_set.shape[0]
        #print(np.sort(features_set.index))

        x_train, x_test, y_train, y_test = train_test_split(features_set, labels_set, test_size=0.1, random_state=42)

        print("TRAIN: ", x_train.shape, y_train.shape)
        print("TEST: ", x_test.shape, y_test.shape)

        # check dimension of the split
        assert x_train.shape[0] + x_test.shape[0] == subset.shape[0]

        # save as csv
        path = output_folder + 'Round' + str(set_number)
        os.mkdir(path)
        x_train.to_csv(output_folder + 'Round' + str(set_number) + '/training_data.csv')
        y_train.to_csv(output_folder + 'Round' + str(set_number) + '/training_labels.csv')
        x_test.to_csv(output_folder + 'Round' + str(set_number) + '/testing_data.csv')
        y_test.to_csv(output_folder + 'Round' + str(set_number) + '/testing_labels.csv')
    

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

        # check there are no leftovers
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
            
        # check there are no leftovers
        assert df_dataset.loc[(df_dataset['n_split_set'] == 0) & (df_dataset['n_label'] == label)].empty == True
        
    return df_dataset
    


create_train_and_test(debthunter_input_file_multiclass, "DatasetD2/", False)



"""x_train, x_test, y_train, y_test = train_test_split(np_features, 
                                                    np_labels, 
                                                    random_state=25, 
                                                    train_size = .90)"""

    
"""print("TRAIN: ", x_train.shape, y_train.shape)
print("TEST: ", x_test.shape, y_test.shape)
print("\n")

print(x_train[:5])"""

"""np.savetxt(output_folder + "training_data.csv", x_train, delimiter=",", fmt="%f")
np.savetxt(output_folder + "training_labels.csv", y_train, delimiter=",")
np.savetxt(output_folder + "testing_data.csv", x_test, delimiter=",", fmt="%f")
np.savetxt(output_folder + "testing_labels.csv", y_test, delimiter=",")"""