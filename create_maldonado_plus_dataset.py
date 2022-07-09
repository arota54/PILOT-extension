import os
import pandas as pd
import csv

path = ""
projects = ["Ant", "ArgoUML", "Columba", "Dubbo", "EMF", "Gradle", "Groovy", "Hibernate", "Hive", "JEdit",
"JFreeChart", "JMeter", "JRuby", "Maven", "Poi", "SpringFramework", "SQuirrel", "Storm", "Tomcat", "Zookeeper"]

# column of the dataframe to create
projects_name = []
labels = []
comments = []

projects_name_zhao = []
labels_zhao = []
comments_zhao = []


def create_and_save_dataframe():
    dataset = pd.DataFrame({
        'projectname': projects_name,
        'classification': labels,
        'commenttext': comments
    })

    dataset.to_csv("datasets/maldonado-plus-dataset.csv", index=False)


# converts "positive" into "SATD" and "negative" into "WITHOUT_CLASSIFICATION"
def convert_labels_text():
    for i in range(len(labels_zhao)):
        if labels_zhao[i] == "negative":
            labels_zhao[i] = "WITHOUT_CLASSIFICATION"
        elif labels_zhao[i] == "positive":
            labels_zhao[i] = "SATD" 
    
    assert labels_zhao.count("WITHOUT_CLASSIFICATION") == (118316 - 5807) and labels_zhao.count("SATD") == 5807


def read_data_and_label_maldonado_only_without_classification():
    with open(path + "datasets/maldonado-dataset.csv", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        #Skip header
        next(csv_reader)

        for row in csv_reader:
            if row[1] == 'WITHOUT_CLASSIFICATION':
                comments.append((row[2]))
                projects_name.append(row[0])
                labels.append(row[1])

    assert len(comments) == 58204 and len(labels) == 58204 and len(projects_name) == 58204


def read_data_and_label_zhao(project):
    data_file_path = path + "ZHAO/data--" + project + ".txt"
    label_file_path = path + "ZHAO/label--" + project + ".txt"

    print(data_file_path)
    with open(data_file_path, encoding="utf8") as data_file:
        data = data_file.read().splitlines()
        comments_zhao.extend(data)

    print(label_file_path)
    with open(label_file_path, encoding="utf8") as label_file:
        label = label_file.read().splitlines()
        labels_zhao.extend(label)

    # check if comments and labels have the same amount of data
    assert len(comments) == len(labels)

    # create a list containing the name of the project with a lenght equal to the amount of data
    projects_name_zhao.extend([project] * len(data))


def append_zhao_satd_to_maldonado():
    indexes_satd = [i for i, item in enumerate(labels_zhao) if item == 'SATD']
    assert len(indexes_satd) == 5807

    for index in indexes_satd:
        comments.append(comments_zhao[index])
        labels.append(labels_zhao[index])
        projects_name.append(projects_name_zhao[index])

    assert len(comments) == 64011 and len(labels) == 64011 and len(projects_name) == 64011


def main(): 
    read_data_and_label_maldonado_only_without_classification()

    for project in projects:
        read_data_and_label_zhao(project)

    # check correct number of comments, labels and project_name
    assert len(comments_zhao) == 118316 and len(labels_zhao) == 118316 and len(projects_name_zhao) == 118316

    convert_labels_text()

    append_zhao_satd_to_maldonado()

    create_and_save_dataframe()

main()