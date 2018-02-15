# For each class extract frames from some videos. Test against imagenet systems
import os
import numpy as np

import config
from meta import load_sound_event_classes, load_videos_info
from FileIO import get_video_filename, get_frame_filename, get_class_directory, write_list_to_csv

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np


def take_frame_from_start(video_info, video_path, output_folder):
	frame_filename = os.path.join(output_folder, get_frame_filename(video_path, "start", config.video_frames_extension))

	ffmpeg_string = "ffmpeg -i {}  -ss 00:00:00.1 -vframes 1 {}".format(video_path, frame_filename)
	os.system(ffmpeg_string)

	return frame_filename


def take_frame_from_middle(video_info, video_path, output_folder):
	frame_filename = os.path.join(output_folder, get_frame_filename(video_path, "middle",  config.video_frames_extension))

	middle_location = str((float(video_info[2]) - float(video_info[1])) / 2)
	
	ffmpeg_string = "ffmpeg -i {}  -ss 00:00:0{} -vframes 1 {}".format(video_path, middle_location, frame_filename)
	os.system(ffmpeg_string)

	return frame_filename


def take_frame_from_end(video_info, video_path, output_folder):
	frame_filename = os.path.join(output_folder, get_frame_filename(video_path, "end",  config.video_frames_extension))

	end_location = str((float(video_info[2]) - float(video_info[1])))

	if(not end_location.startswith("10.")):
		end_location = "0" + end_location
	
	ffmpeg_string = "ffmpeg -i {}  -ss 00:00:{} -vframes 1 {}".format(video_path, end_location, frame_filename)
	os.system(ffmpeg_string)

	return frame_filename


def take_n_equal_spaced_frames(num_frames, video_info, video_path, output_folder):
	filename = os.path.splitext(os.path.basename(video_path))[0] + "_spaced_%d" + config.video_frames_extension
	frame_filename = os.path.join(output_folder, filename)

	video_length = float(video_info[2]) - float(video_info[1])
	fps = num_frames / video_length

	ffmpeg_string = "ffmpeg -i {} -vf fps={} {}".format(video_path, fps, frame_filename)
	os.system(ffmpeg_string)

	return frame_filename # Going to have to handle this differently because of the %d


def take_frame_each_scene(video_info, video_filename, video_location, output_folder):
	pass


def load_model():
	model = InceptionV3(weights='imagenet')
	return model


def test_frame(model, path):
	img = image.load_img(path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	preds = model.predict(x)
	# decode the results into a list of tuples (class, description, probability)
	# (one such list for each sample in the batch)
	print('Predicted:', decode_predictions(preds, top=3)[0])



class_labels = load_sound_event_classes()
videos_by_classes = load_videos_info(config.video_training_data_location, config.training_data_csv_file)
#model = load_model()


videos_with_errors = list()

for class_label, class_name in class_labels.items():
	output_folder = get_class_directory(config.video_training_frames_location, class_name)

	if not os.path.exists(output_folder):
			os.makedirs(output_folder)

	for video_info in videos_by_classes[class_label]:
		video_filename = get_video_filename(video_info[0], video_info[1], video_info[2], config.video_file_extension)
		video_path = os.path.join(config.video_training_data_location, video_filename)

		if os.path.exists(video_path):
			frame_filename = take_frame_from_start(video_info, video_path, output_folder)

			if os.path.exists(frame_filename):
				#test_frame(model, frame_filename)
				os.remove(frame_filename)
			else:
				videos_with_errors.append(video_info)

print(len(videos_with_errors))
write_list_to_csv("training-videos-with-errors.csv", videos_with_errors)
