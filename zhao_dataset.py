# create zhao dataset csv (118316)
# create zhao-g multiclass dataset arff for debthunter to label comments (2995)
# create zhao multiclass containing maldonado multiclass (4071) and zhao-g multiclass (2995)
import os
import pandas as pd
import arff
import csv
import numpy as np
import string
import json
from scipy.io import arff as scyarff
  
path = "ZHAO/"
projects = ["Ant", "ArgoUML", "Columba", "Dubbo", "EMF", "Gradle", "Groovy", "Hibernate", "Hive", "JEdit",
"JFreeChart", "JMeter", "JRuby", "Maven", "Poi", "SpringFramework", "SQuirrel", "Storm", "Tomcat", "Zookeeper"]
ten_projects_zhao = ["Dubbo", "Gradle", "Groovy", "Hive", "Maven", "Poi", "SpringFramework", "Storm", "Tomcat", "Zookeeper"]

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# column of the dataframe to create
projects_name = []
labels = []
comments = []


def create_and_save_dataframe():    
    dataset = pd.DataFrame({
        'projectname': projects_name,
        'classification': labels,
        'commenttext': comments
    })

    dataset.to_csv("datasets/zhao-dataset.csv", index=False)

    return dataset

# converts "positive" into "SATD" and "negative" into "WITHOUT_CLASSIFICATION"
def convert_labels_text():
    for i in range(len(labels)):
        if labels[i] == "negative":
            labels[i] = "WITHOUT_CLASSIFICATION"
        elif labels[i] == "positive":
            labels[i] = "SATD" 
    
    assert labels.count("WITHOUT_CLASSIFICATION") == (118316 - 5807) and labels.count("SATD") == 5807

def read_data_and_label(project):
    data_file_path = path + "data--" + project + ".txt"
    label_file_path = path + "label--" + project + ".txt"

    #print(data_file_path)
    with open(data_file_path, encoding="utf8") as data_file:
        data = data_file.read().splitlines()
        comments.extend(data)

    #print(label_file_path)
    with open(label_file_path, encoding="utf8") as label_file:
        label = label_file.read().splitlines()
        labels.extend(label)

    # check if comments and labels have the same amount of data
    assert len(comments) == len(labels)

    # create a list containing the name of the project with a lenght equal to the amount of data
    projects_name.extend([project] * len(data))

def remove_non_satd(dataset):
    dataset = dataset[dataset.classification == "SATD"]
    assert dataset.shape[0] == 5807
    
    return dataset

def save_zhao_g_arff(dataset):
    dataset.insert(1, "package", dataset.projectname)
    dataset.insert(2, "top_package", dataset.projectname)

    dataset_no_classification = dataset.drop('classification', axis=1)

    # 5087 comments
    """arff.dump('datasets/whole-zhao-multiclass.arff'
      , dataset_no_classification.values
      , relation='comments'
      , names=["projectname", "package", "top_package", "comment"])"""

    
    zhao_2021_g = dataset_no_classification[dataset_no_classification.projectname.isin(ten_projects_zhao)]
    assert zhao_2021_g.shape[0] == 2995

    # input for DebtHunter
    arff.dump('datasets/zhao-g-multiclass-to-be-labeled.arff'
      , zhao_2021_g.values
      , relation='comments'
      , names=["projectname", "package", "top_package", "comment"])

    return zhao_2021_g

def read_data_and_label_maldonado_only_satd():
    comments_maldonado = []
    labels_maldonado = []
    projects_name_maldonado = []
    with open(path + "../datasets/maldonado-dataset.csv", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        #Skip header
        next(csv_reader)

        for row in csv_reader:
            if row[1] != 'WITHOUT_CLASSIFICATION':
                comments_maldonado.append((row[2]))
                projects_name_maldonado.append(row[0])
                labels_maldonado.append(row[1])

    assert len(comments_maldonado) == 4071 and len(labels_maldonado) == 4071 and len(projects_name_maldonado) == 4071

    maldonado = pd.DataFrame({
        'projectname': projects_name_maldonado,
        'classification': labels_maldonado,
        'commenttext': comments_maldonado
    })

    return maldonado

def get_comment(array):
    #print(array)
    del array[0:3]
    del array[len(array)-1]
    #print(array)

    comment = ""
    for item in array:
        comment += item
    """print(comment)
    print()"""
    return comment

def read_data_and_label_zhao_g():
    comments_zhao = []
    labels_zhao = []
    projects_name_zhao = []
    with open(path + "../datasets/zhao-g-multiclass-labeled.csv", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        #Skip header
        next(csv_reader)

        for row in csv_reader:
            projects_name_zhao.append(row[0])
            labels_zhao.append(row[len(row)-1])
            comment = get_comment(row)
            comments_zhao.append(comment)
            
    assert len(comments_zhao) == 2995 and len(labels_zhao) == 2995 and len(projects_name_zhao) == 2995

    zhao_g = pd.DataFrame({
        'projectname': projects_name_zhao,
        'classification': labels_zhao,
        'commenttext': comments_zhao
    })

    #print(zhao_g.head())

    """df = zhao_g.groupby(['classification'], sort=False).size().reset_index(name='Count')
    print (df)"""

    return zhao_g

def create_and_save_pilot_multiclass(zhao_g, maldonado):
    new_dataset = pd.concat([zhao_g, maldonado], axis=0)
    new_dataset.to_csv("datasets/pilot-dataset-multiclass.csv", index=False)



    data = new_dataset.drop(labels=["projectname"], axis=1)
    data = data[['commenttext','classification']]

    # input for DebtHunter
    arff.dump('datasets/pilot-dataset.arff'
      , data.values
      , relation='comments'
      , names=["comment", "classification"])

def main(): 
    for project in projects:
        read_data_and_label(project)

    # check correct number of comments, labels and project_name
    assert len(comments) == 118316 and len(labels) == 118316 and len(projects_name) == 118316

    convert_labels_text()
    zhao = create_and_save_dataframe()
    zhao_g = remove_non_satd(zhao)
    save_zhao_g_arff(zhao_g)

    maldonado_multiclass = read_data_and_label_maldonado_only_satd()
    zhao_g_multiclass = read_data_and_label_zhao_g()
    create_and_save_pilot_multiclass(zhao_g_multiclass, maldonado_multiclass)

main()