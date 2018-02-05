# For each class extract frames from some videos. Test against imagenet systems
import os
import FileIO
import numpy as np
import pandas as pd
from config import *
from collections import defaultdict


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


def get_video_filename(video_info, video_extension):
	return "{}_{}_{}{}".format(video_info[0], video_info[1], video_info[2], video_extension)



def create_frame_filename():
	pass



def make_frame_folders_for_classes(class_labels, frames_folder):
	for class_label in class_labels:
		path = os.path.join(frames_folder, class_label)

		if not os.path.exists(path):
			os.makedirs(path)


class_labels = load_sound_event_classes()
videos_by_classes = load_videos_info(video_training_data_location, training_data_csv_file)
# make_frame_folders_for_classes(class_labels, video_training_frames_location)

for class_label in class_labels:
	print(class_labels[class_label])

	for video_info in videos_by_classes[class_label]:
		video_filename = get_video_filename(video_info, video_file_extension)


#os.system('ffmpeg -i input.flv -ss 00:00:14.435 -vframes 1 out.png')