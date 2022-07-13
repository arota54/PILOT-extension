import os
import pandas as pd
import arff
  
path = "ZHAO/"
projects = ["Ant", "ArgoUML", "Columba", "Dubbo", "EMF", "Gradle", "Groovy", "Hibernate", "Hive", "JEdit",
"JFreeChart", "JMeter", "JRuby", "Maven", "Poi", "SpringFramework", "SQuirrel", "Storm", "Tomcat", "Zookeeper"]

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

    print(data_file_path)
    with open(data_file_path, encoding="utf8") as data_file:
        data = data_file.read().splitlines()
        comments.extend(data)

    print(label_file_path)
    with open(label_file_path, encoding="utf8") as label_file:
        label = label_file.read().splitlines()
        labels.extend(label)

    # check if comments and labels have the same amount of data
    assert len(comments) == len(labels)

    # create a list containing the name of the project with a lenght equal to the amount of data
    projects_name.extend([project] * len(data))

def remove_non_satd(dataset):
    print(dataset.shape)
    dataset = dataset[dataset.classification == "SATD"]
    print(dataset.shape)


def main(): 
    for project in projects:
        read_data_and_label(project)

    # check correct number of comments, labels and project_name
    assert len(comments) == 118316 and len(labels) == 118316 and len(projects_name) == 118316

    convert_labels_text()
    dataset = create_and_save_dataframe()
    remove_non_satd(dataset)


main()