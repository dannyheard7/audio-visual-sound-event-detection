import pandas as pd
from collections import defaultdict

import FileIO
import re

def load_sound_event_classes():
    df = pd.read_csv("metadata/sound_event_class_list.csv")
    # class_labels = df.iloc[:,2].values.tolist()
    class_labels = df.iloc[:,[0,2]].set_index('Class_ID').T.to_dict('records')[0]

    return class_labels


def load_videos_info_by_class(csv_file):
    data_set = FileIO.load_csv_as_list(csv_file)
    videos_dict =  defaultdict(list)

    for row in data_set:
        video_class_id = row[4]
        
        video_class_labels = re.findall('[A-Z][^A-Z]*', row[3])
        video_class_labels = [x[:-1] if x.endswith(",") else x for x in video_class_labels]

        if len(video_class_labels) > 1:
            for x in video_class_labels:
                videos_dict[x].append(row[0:3])
        else:
            videos_dict[video_class_labels[0]].append(row[0:3])

    return videos_dict
