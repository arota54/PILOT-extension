import csv
import string
import pandas as pd
import tensorflow as tf
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import os, shutil
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
# Activate TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))

# utilizzo di una GPU su scheda grafica locale
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='File .csv containing the dataset')
args = parser.parse_args()
input_file = args.input_file"""

base_dir = ''

maldonado_input_file = base_dir + 'datasets/maldonado-dataset.csv'
maldonado_features_matrices_binary = base_dir + 'tf-idf/features-matrices/features-matrices-maldonado-binary.csv'
maldonado_features_matrices_multiclass = base_dir + 'tf-idf/features-matrices/features-matrices-maldonado-multiclass.csv'


debthunter_input_file = base_dir + 'datasets/debthunter-dataset.csv'
debthunter_features_matrices_binary = base_dir + 'tf-idf/features-matrices/features-matrices-debthunter-binary.csv'
debthunter_features_matrices_multiclass = base_dir + 'tf-idf/features-matrices/features-matrices-debthunter-multiclass.csv'


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
                result = result + ' ' + stemmer.stem(wordnet_lemmatizer.lemmatize(text))

    #print(result)
    return result


def read_comments_and_labels(): 
    comments = []
    projects_name = []
    classifications = []


    with open(INPUT_FILE) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        #Skip header
        next(csv_reader)
        for row in csv_reader:
            
            if BINARY_CLASSIFICATION or row[1] != 'WITHOUT_CLASSIFICATION':
                if DATASET:
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


def tf_idf(comments, classifications):
    vect = TfidfVectorizer()
    vects = vect.fit_transform(comments)

    # Select all rows from the data set
    td = pd.DataFrame(vects.todense()).iloc[:]

    # td.columns contiene le feature (parole prese da comments)
    td.columns = vect.get_feature_names_out()

    # term_document_matrix contiene coppie (parola, peso) per ogni parola unica presa da tutti i commenti
    # il peso è 0 se la parola non è presente nel commento
    term_document_matrix = td.T
    term_document_matrix.columns = [str(i) + ' - ' + classifications[i - 1] for i in range(1, len(comments) + 1)]

    if BINARY_CLASSIFICATION:
        term_document_matrix['total_count'] = term_document_matrix.astype(bool).sum(axis=1)
        term_document_matrix = term_document_matrix.sort_values(by ='total_count',ascending=False)[:]
        term_document_matrix = term_document_matrix.loc[term_document_matrix['total_count'] >= 4]
        term_document_matrix = term_document_matrix.drop(columns=['total_count'])

    if BINARY_CLASSIFICATION:
        term_document_matrix.T.to_csv(OUTPUT_FILE_BINARY)
    else:
        term_document_matrix.T.to_csv(OUTPUT_FILE_MULTICLASS)


def main(binary_classification, dataset) :
    global BINARY_CLASSIFICATION, DATASET, MAX_LEN, INPUT_FILE, OUTPUT_FILE_BINARY, OUTPUT_FILE_MULTICLASS

    BINARY_CLASSIFICATION = binary_classification
    DATASET = dataset

    if DATASET:
        INPUT_FILE = maldonado_input_file
        OUTPUT_FILE_BINARY = maldonado_features_matrices_binary
        OUTPUT_FILE_MULTICLASS = maldonado_features_matrices_multiclass
    else:
        INPUT_FILE = debthunter_input_file
        OUTPUT_FILE_BINARY = debthunter_features_matrices_binary
        OUTPUT_FILE_MULTICLASS = debthunter_features_matrices_multiclass

    print("Read comments and labels")
    comments, labels = read_comments_and_labels() 

    print("TD_IDF")
    tf_idf(comments, labels)


# BINARY_CLASSIFICATION = True prende l'intero dataset (WITHOUT_CLASSIFICATION + tutte le altre etichette)
# BINARY_CLASSIFICATION = False prende solo le altre etichette (non prende WITHOUT_CLASSIFICATION)
# DATASET = True (Maldonado), DATASET = False (DebtHunter)
#main(True, False) # DebtHunter binary
#main(binary_classification=True, dataset=True) # Maldonado binary
main(binary_classification=False, dataset=True) # Maldonado multiclass