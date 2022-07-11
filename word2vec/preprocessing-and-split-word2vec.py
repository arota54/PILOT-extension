from cgi import test
import csv
import string
import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing as kprocessing
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api
import os, shutil
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from gensim import models
import time

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))

#w2v = api.load("word2vec-google-news-300")

# utilizzo di una GPU su scheda grafica locale
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

start_time = time.time()
base_dir = ''

w2v_path = base_dir + "word2vec/model/"

maldonado_input_file = base_dir + 'datasets/maldonado-dataset.csv'
maldonado_output_file_binary = base_dir + 'word2vec/binary/DatasetD1/df-maldonado-binary.csv'
maldonado_output_file_multiclass = base_dir + 'word2vec/multiclass/DatasetD1/df-maldonado-multiclass.csv'
maldonado_output_folder_multiclass = base_dir + 'word2vec/multiclass/DatasetD1/'
maldonado_output_folder_binary = base_dir + 'word2vec/binary/DatasetD1/'

zhao_input_file = base_dir + 'datasets/zhao-dataset.csv'
zhao_output_file_binary = base_dir + 'word2vec/binary/DatasetZhao/df-zhao-binary.csv'
zhao_output_folder_binary = base_dir + 'word2vec/binary/DatasetZhao/'

maldonadoplus_input_file = base_dir + 'datasets/maldonado-plus-dataset.csv'
maldonadoplus_output_file_binary = base_dir + 'word2vec/binary/DatasetMaldonadoPlus/df-maldonadoplus-binary.csv'
maldonadoplus_output_folder_binary = base_dir + 'word2vec/binary/DatasetMaldonadoPlus/'

debthunter_input_file = base_dir + 'datasets/debthunter-dataset.csv'
debthunter_output_file_binary = base_dir + 'word2vec/binary/DatasetD2/df-debthunter-binary.csv'
debthunter_output_file_multiclass = base_dir + 'word2vec/multiclass/DatasetD2/df-debthunter-multiclass.csv'
debthunter_output_folder_multiclass = base_dir + 'word2vec/multiclass/DatasetD2/'
debthunter_output_folder_binary = base_dir + 'word2vec/binary/DatasetD2/'


def clean_term(text):
    text = text.lower()
    return "".join(char for char in text
                   if char not in string.punctuation)


def standardize(text):
    stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    nltk_tokens = nltk.word_tokenize(text)
    
    result = ''
    for w in nltk_tokens:
        if w not in stop_words:
            text = clean_term(w)
            # if text is not a number
            if not text.isdigit():
                # text = text.replace("'", "")
                result = result + ' ' + stemmer.stem(wordnet_lemmatizer.lemmatize(text))

    #print(result)
    return result


def remove_single_quotes(comment):
    res = comment.split()
    string = ""

    for word in res:
        if word != "'":
            word = ''.join(word.strip("'"))
            string = string + " " + word 

    return string
    


def read_comments_and_labels(): 
    comments = []
    projects_name = []
    classifications = []

    print("Read comments and labels")
    with open(INPUT_FILE, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        #Skip header
        next(csv_reader)
        for row in csv_reader:
            
            if BINARY_CLASSIFICATION or row[1] != 'WITHOUT_CLASSIFICATION':
                if DATASET == 1 or DATASET == 3 or DATASET == 4:
                    comments.append(standardize(row[2]))
                    projects_name.append(row[0])
                    classifications.append(row[1])
                else:
                    #string = remove_single_quotes(row[0])
                    #print(string)
                    comments.append(string)
                    classifications.append(row[1])

    return comments, classifications


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


def create_dataframe(comments, labels):
    labels_int = labels_from_text_to_int(labels)

    df = pd.DataFrame(comments, columns=["comments"]).iloc[:]
    df.insert(df.shape[1], 'labels', labels_int) 


    print("Start to save df on csv file")
    if BINARY_CLASSIFICATION :
        df.to_csv(OUTPUT_FILE_BINARY, index=False)
    else :
        df.to_csv(OUTPUT_FILE_MULTICLASS, index=False)
    print("Df saved on csv file")

    return df

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


def word2vec(x_train, x_validation, x_test, output_folder, tokenizer, matrix):
    print("Start word2vec")

    train_sequences = kprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=MAX_LEN)
    validation_sequences = kprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(x_validation), maxlen=MAX_LEN)
    test_sequences = kprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=MAX_LEN)

    print(train_sequences.shape, validation_sequences.shape, test_sequences.shape)
    #print(x_train)

    np.savetxt(output_folder + '/emb_matrix.csv', matrix, delimiter=",")
    np.savetxt(output_folder + '/train_sequences.csv', train_sequences, delimiter=",")
    np.savetxt(output_folder + '/validation_sequences.csv', validation_sequences, delimiter=",")
    np.savetxt(output_folder + '/test_sequences.csv', test_sequences, delimiter=",")
    print("End word2vec")


def word2vec_tokenizer_matrix(comments):
    print("Start word2vec")
    max_words = 5097
    tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', num_words=max_words, oov_token="<pad>", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(comments)
    voc = tokenizer.word_index
    reverse_voc = dict([(value, key) for (key, value) in voc.items()])

    #print(reverse_voc)

    w2v = models.KeyedVectors.load_word2vec_format(w2v_path + 
    'GoogleNews-vectors-negative300.bin.gz', binary=True)

    # Build weights of the embbeddings matrix using w2v
    emb_matrix=np.zeros((max_words+1, 300))
    for i in range(max_words):
        w = reverse_voc[i+1]
        if w in w2v:
            emb_matrix[i+1,:] = w2v[w]
    emb_size = emb_matrix.shape[1]
    print(emb_matrix.shape)

    #np.savetxt(output_folder + '/emb_matrix.csv', emb_matrix, delimiter=",")
    return tokenizer, emb_matrix


def main(binary_classification, dataset) :
    global BINARY_CLASSIFICATION, DATASET, MAX_LEN, INPUT_FILE, OUTPUT_FILE_BINARY, OUTPUT_FILE_MULTICLASS, OUTPUT_FOLDER_BINARY, OUTPUT_FOLDER_MULTICLASS

    BINARY_CLASSIFICATION = binary_classification
    DATASET = dataset

    if DATASET == 1:
        INPUT_FILE = maldonado_input_file
        OUTPUT_FILE_BINARY = maldonado_output_file_binary
        OUTPUT_FILE_MULTICLASS = maldonado_output_file_multiclass
        OUTPUT_FOLDER_BINARY = maldonado_output_folder_binary
        OUTPUT_FOLDER_MULTICLASS = maldonado_output_folder_multiclass
    elif DATASET == 2:
        INPUT_FILE = debthunter_input_file
        OUTPUT_FILE_BINARY = debthunter_output_file_binary
        OUTPUT_FILE_MULTICLASS = debthunter_output_file_multiclass
        OUTPUT_FOLDER_BINARY = debthunter_output_folder_binary
        OUTPUT_FOLDER_MULTICLASS = debthunter_output_folder_multiclass
    elif DATASET == 3:
        INPUT_FILE = zhao_input_file
        OUTPUT_FILE_BINARY = zhao_output_file_binary
        OUTPUT_FOLDER_BINARY = zhao_output_folder_binary
    elif DATASET == 4:
        INPUT_FILE = maldonadoplus_input_file
        OUTPUT_FILE_BINARY = maldonadoplus_output_file_binary
        OUTPUT_FOLDER_BINARY = maldonadoplus_output_folder_binary

    if BINARY_CLASSIFICATION:
        output_folder = OUTPUT_FOLDER_BINARY
    else:
        output_folder = OUTPUT_FOLDER_MULTICLASS
    delete_folder(output_folder)

    comments, labels = read_comments_and_labels() 
    #MAX_LEN = len(max(comments, key=len))
    #MAX_LEN = 2000
    MAX_LEN = 5097
    print(MAX_LEN)

    df = create_dataframe(comments, labels)

    df_comments, df_labels = df['comments'], df['labels']
    print(df_comments.shape, df_labels.shape)

    # save dataframes into 10 folders (Round1, ...)
    set_number = 1
    kf = KFold(n_splits=10, shuffle=True, random_state=42, )
    print(kf)

    x, y = np.array(df_comments), np.array(df_labels)
    print("WHOLE: ", x.shape, y.shape)
    unique_elements, counts_elements = np.unique(y, return_counts=True)
    print(np.asarray((unique_elements, counts_elements)))

    tokenizer, matrix = word2vec_tokenizer_matrix(df_comments)

    for train_val_index, test_index in kf.split(df_comments, df_labels):
        print("\nSET: ", set_number)

        x_train_val, y_train_val = df_comments.iloc[train_val_index], df_labels.iloc[train_val_index] 
        x_test, y_test = df_comments.iloc[test_index], df_labels.iloc[test_index] 

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
        assert x_train.shape[0] + x_validation.shape[0] + x_test.shape[0] == df_comments.shape[0]
        if BINARY_CLASSIFICATION :
            val = (0 in y_train.to_numpy() and 1 in y_train.to_numpy())
            assert val == True
            val = (0 in y_validation.to_numpy() and 1 in y_validation.to_numpy())
            assert val == True
            val = (0 in y_test.to_numpy() and 1 in y_test.to_numpy())
            assert val == True
        else :
            """val = (0 in y_train and 1 in y_train and 2 in y_train and 3 in y_train and 4 in y_train)
            assert val == True
            val = (0 in y_test and 1 in y_test and 2 in y_test and 3 in y_test and 4 in y_test)
            assert val == True"""
        
        
        path = output_folder + 'Round' + str(set_number)
        os.mkdir(path)   
        word2vec(x_train, x_validation, x_test, path, tokenizer, matrix)

        print(x_train)

        # save train, validation and test as csv
        #np.savetxt(path + '/training_data.csv', x_train, delimiter=",", fmt="%s")
        np.savetxt(path + '/training_labels.csv', y_train.astype(int), delimiter=",", fmt="%i")
        #np.savetxt(path + '/validation_data.csv', x_validation, delimiter=",", fmt="%s")
        np.savetxt(path + '/validation_labels.csv', y_validation.astype(int), delimiter=",", fmt="%i")
        #np.savetxt(path + '/testing_data.csv', x_test, delimiter=",", fmt="%s")
        np.savetxt(path + '/testing_labels.csv', y_test.astype(int), delimiter=",", fmt="%i")

        set_number = set_number + 1
        print("Seconds: ", (time.time() - start_time)) 


# BINARY_CLASSIFICATION = True prende l'intero dataset (WITHOUT_CLASSIFICATION + tutte le altre etichette)
# BINARY_CLASSIFICATION = False prende solo le altre etichette (non prende WITHOUT_CLASSIFICATION)
# DATASET = 1 (Maldonado), DATASET = 2 (DebtHunter), DATASET = 3 (Zhao), DATASET = 3 (Maldonado Plus)
#main(True, False) # DebtHunter binary
#main(True, 1) # Maldonado binary
#main(True, 3) # Zhao binary
#main(False, 1)
main(False, 1) 