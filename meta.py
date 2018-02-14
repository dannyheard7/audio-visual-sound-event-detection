import pandas as pd
from collections import defaultdict

import FileIO

def load_sound_event_classes():
	df = pd.read_csv("metadata/sound_event_class_list.csv")
	# class_labels = df.iloc[:,2].values.tolist()
	class_labels = df.iloc[:,[0,2]].set_index('Class_ID').T.to_dict('records')[0]

	return class_labels
	

def load_videos_info(videos_location, csv_file):
	data_set = FileIO.load_csv_as_list(csv_file)
	videos_dict =  defaultdict(list)

	for row in data_set:
		video_class_id = row[4]

		if "," in video_class_id:
			for x in video_class_id.split(','):
				videos_dict[x].append(row[0:3])
		else:
			videos_dict[video_class_id].append(row[0:3])


	return videos_dict
