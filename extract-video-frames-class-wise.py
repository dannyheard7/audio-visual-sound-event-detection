# For each class extract frames from some videos. Test against imagenet systems
import os
import FileIO
import numpy as np
import pandas as pd
import config
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
	return "Y{}_{}_{}{}".format(video_info[0], video_info[1], video_info[2], video_extension) # Download video script places a Y at the start of the filename


def create_frame_filename(video_filename, frame_type):
	filename, _ = os.path.splitext(video_filename)
	filename += "_{}{}".format(frame_type, config.video_frames_extension)
	return filename


def take_frame_from_start(video_info, video_filename, video_location, output_folder):
	frame_filename = os.path.join(output_folder, create_frame_filename(video_filename, "start"))
	video_filename = os.path.join(video_location, video_filename)

	ffmpeg_string = "ffmpeg -i {}  -ss 00:00:00.1 -vframes 1 {}".format(video_filename, frame_filename)
	os.system(ffmpeg_string)


def take_frame_from_middle(video_info, video_filename, video_location, output_folder):
	frame_filename = os.path.join(output_folder, create_frame_filename(video_filename, "middle"))
	video_filename = os.path.join(video_location, video_filename)

	middle_location = str((float(video_info[2]) - float(video_info[1])) / 2)
	
	# All videos are 10s or shorter
	#if(not middle_location.startswith("10.")):
	#	middle_location = "0" + middle_location

	ffmpeg_string = "ffmpeg -i {}  -ss 00:00:0{} -vframes 1 {}".format(video_filename, middle_location, frame_filename)
	os.system(ffmpeg_string)


def take_frame_from_end(video_info, video_filename, video_location, output_folder):
	frame_filename = os.path.join(output_folder, create_frame_filename(video_filename, "end"))
	video_filename = os.path.join(video_location, video_filename)

	end_location = str((float(video_info[2]) - float(video_info[1])))

	if(not end_location.startswith("10.")):
		end_location = "0" + end_location
	
	ffmpeg_string = "ffmpeg -i {}  -ss 00:00:{} -vframes 1 {}".format(video_filename, end_location, frame_filename)
	os.system(ffmpeg_string)


def take_n_equal_spaced_frames(num_frames, video_info video_filename, video_location, output_folder):
	video_filename = os.path.join(video_location, video_filename)

	filename = os.path.splitext(video_filename)[0] + "_spaced_%d" + config.video_frames_extension
	frame_filename = os.path.join(output_folder, filename)
	print(frame_filename)

	video_length = float(video_info[2]) - float(video_info[1])

	fps = 

	ffmpeg_string = "ffmpeg -i {} -vf fps=1 {}".format(video_filename, frame_filename)

def create_and_populate_csv():
	pass


class_labels = load_sound_event_classes()
videos_by_classes = load_videos_info(config.video_training_data_location, config.training_data_csv_file)


for class_label, class_name in class_labels.items():
	normalised_class_name = class_name.replace(" ", "-").replace(',', '').lower()
	output_folder = os.path.join(config.video_training_frames_location, normalised_class_name)

	if not os.path.exists(output_folder):
			os.makedirs(output_folder)

	for video_info in videos_by_classes[class_label]:
		video_filename = get_video_filename(video_info, config.video_file_extension)
		take_n_equal_spaced_frames(1, video_info, video_filename,  config.video_training_data_location, output_folder)


#os.system('ffmpeg -i input.flv -ss 00:00:14.435 -vframes 1 out.png')