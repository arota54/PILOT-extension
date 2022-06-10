import csv
from email.mime import base
import string
import pandas as pd
import tensorflow as tf
import argparse
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Activate TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import _fake_quantize_learnable_per_tensor_affine

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))

"""parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='File .csv containing the dataset')
args = parser.parse_args()
input_file = args.input_file"""

base_dir = ''
maldonado_input_file = base_dir + 'datasets/maldonado-dataset.csv'
maldonado_output_file_binary = base_dir + 'features-matrices/maldonado-features-matrices-binary.csv'
maldonado_output_file_multiclass = base_dir + 'features-matrices/maldonado-features-matrices-multiclass.csv'

debthunter_input_file = base_dir + 'datasets/debthunter-dataset.csv'
debthunter_output_file_binary = base_dir + 'features-matrices/debthunter-features-matrices-binary.csv'
debthunter_output_file_multiclass = base_dir + 'features-matrices/debthunter-features-matrices-multiclass.csv'


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
    return result


# BINARY_CLASSIFICATION = True prende l'intero dataset (WITHOUT_CLASSIFICATION + tutte le altre etichette
# BINARY_CLASSIFICATION = False prende solo le altre etichette (non prende WITHOUT_CLASSIFICATION)
# DATASET = True (Maldonado), DATASET = False (DebtHunter)
def create_feature_matrices(BINARY_CLASSIFICATION, DATASET): 
    comments = []
    projects_name = []
    classifications = []
    BINARY_CLASSIFICATION = BINARY_CLASSIFICATION

    if DATASET:
        INPUT_FILE = maldonado_input_file
        OUTPUT_FILE_BINARY = maldonado_output_file_binary
        OUTPUT_FILE_MULTICLASS = maldonado_output_file_multiclass
    else:
        INPUT_FILE = debthunter_input_file
        OUTPUT_FILE_BINARY = debthunter_output_file_binary
        OUTPUT_FILE_MULTICLASS = debthunter_output_file_multiclass


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
                    comments.append(row[0])
                    classifications.append(row[1])

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

    print(term_document_matrix.T.shape)


create_feature_matrices(True, True) # Maldonado binary
create_feature_matrices(False, True) # Maldonado multiclass
create_feature_matrices(True, False) # DebtHunter binary
create_feature_matrices(False, False) # DebtHunter multiclass
