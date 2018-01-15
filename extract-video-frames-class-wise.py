# For each class extract frames from some videos. Test against imagenet systems
import os
import FileIO
import numpy as np
import pandas as pd
from config import *


def load_sound_event_classes():
	df = pd.read_csv("metadata/sound_event_class_list.csv")
	# class_labels = df.iloc[:,2].values.tolist()
	class_labels = df.iloc[:,[0,2]].set_index('Class_ID').T.to_dict('records')[0]

	return class_labels


def load_videos_info(videos_location, csv_file):
	data_set = FileIO.load_csv_as_list(csv_file)
	videos_dict = dict()

	for row in data_set:
		video_class_id = row[4]
		#print(video_class_id)

		#if "," in video_class_id:
			# iterate over all class_ids
			#pass
	
		if video_class_id in videos_dict:
			videos_dict[video_class_id].append(row[0:3])
		else:
			videos_dict = {video_class_id: [row[0:3]]}

	return videos_dict



class_labels = load_sound_event_classes()


videos = load_videos_info(video_training_data_location, training_data_csv_file)

for key in videos:
	print(key)

#os.system('ffmpeg -i input.flv -ss 00:00:14.435 -vframes 1 out.png')