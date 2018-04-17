import pandas as pd
from collections import defaultdict

import FileIO
import re

import config
import numpy as np

def load_sound_event_classes():
    df = pd.read_csv("metadata/sound_event_class_list.csv")
    # class_labels = df.iloc[:,2].values.tolist()
    class_labels = df.iloc[:,[0,2]].set_index('Class_ID').T.to_dict('records')[0]

    return class_labels


def load_videos_info_by_class(csv_file):
    dataset = FileIO.load_csv_as_list(csv_file)
    videos_dict =  defaultdict(list)

    for row in dataset:
        video_class_labels = re.findall('[A-Z][^A-Z]*', row[3])
        video_class_labels = [x[:-1] if x.endswith(",") else x for x in video_class_labels]

        if len(video_class_labels) > 1:
            for x in video_class_labels:
                videos_dict[x].append(row[0:3])
        else:
            videos_dict[video_class_labels[0]].append(row[0:3])

    return videos_dict


def get_labels(filenames, csv_file):
    dataset = FileIO.load_csv_as_list(csv_file)
    labels_list = []

    ids_sorted = [x for _, x in sorted(zip(config.labels, config.ids), key=lambda pair: pair[0])]

    for row in dataset:
        video_id = row[0]
        #full_filename = next(filter(lambda x: video_id.encode('utf-8') in x, filenames), None)
        full_filename = (video_id + "_" + row[1] + "_" + row[2] + ".wav").encode('utf-8')
        
        if full_filename in filenames:
            video_class_ids = row[4].split(",")

            # These need to be a list of 1s/0s, 1 if the file has the class at that index or 0 otherwise
            video_class_labels = [1 if class_id in video_class_ids else 0 for class_id in ids_sorted]
            labels_list.append((full_filename, video_class_labels))

    labels_list = sorted(labels_list, key=lambda x: filenames.index(x[0]))
    labels_list = [labels for filename, labels in labels_list]

    return labels_list


def get_images_labels(csv_file):
    dataset = FileIO.load_csv_as_list(csv_file)
    labels_list = {}
    sorted_labels = sorted(config.labels, key=str.lower)

    for row in dataset:
        video_id = row[0]
        full_filename = "Y" + video_id + "_" + row[1] + "_" + row[2] + "_middle.png"

        video_class_labels = re.findall('[A-Z][^A-Z]*', row[3])
        video_class_labels = [x[:-1] if x.endswith(",") else x for x in video_class_labels]

        video_class_labels = [1 if label in video_class_labels else 0 for label in sorted_labels]

        # These need to be a list of 1s/0s, 1 if the file has the class at that index or 0 otherwise
        labels_list[full_filename] = np.asarray(video_class_labels)

    return labels_list


def get_train_labels_list():
    videos_by_class = load_videos_info_by_class(config.training_data_csv_file)
    keys = sorted(list(videos_by_class.keys()), key=str.lower)
    labels = []

    for label in keys:
        labels.extend([label] * len(videos_by_class[label]))

    return labels
