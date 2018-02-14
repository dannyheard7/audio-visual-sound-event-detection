import csv
import os

def get_video_filename(video_id, start_time, end_time, video_extension):
	return "Y{}_{}_{}{}".format(video_id, start_time, end_time, video_extension) # Download video script places a Y at the start of the filename


def get_frame_filename(video_path, frame_type, frames_extension):
	filename, _ = os.path.splitext(os.path.basename(video_path))
	filename += "_{}{}".format(frame_type, frames_extension)
	return filename


def get_class_directory(location, class_name):
	normalised_class_name = class_name.replace(" ", "-").replace(',', '').lower()
	output_folder = os.path.join(location, normalised_class_name)
	return output_folder


def load_csv_as_list(filename):
        with open(filename, 'rt') as csvfile:
                data = csv.reader(csvfile)

                csv_list = [row for row in data]

        return csv_list


def extract_evaluation_video_ids():
	with open('evaluation_set.csv', 'w') as out, open('metadata/groundtruth_strong_label_evaluation_set.csv', 'r') as in_file:
	    for row in csv.reader(in_file):
	        audio_file_name = row[0]
	        split_dot = audio_file_name.split(".")[0]
	        k = split_dot.rfind("_")
	        video_id = split_dot[:k]

	        wav_pos =  audio_file_name.rfind(".wav")
	        last_underscore = audio_file_name.rfind("_") 
	        end_time = audio_file_name[(last_underscore + 1):wav_pos].rstrip()

	        audio_file_name_split = audio_file_name[:last_underscore]
	        second_last_underscore = audio_file_name_split.rfind("_")

	        start_time =  audio_file_name_split[(second_last_underscore + 1):last_underscore]

	        classes = row[-1]
	        
	        print(video_id + ',' + start_time + ',' + end_time + ',' + classes, file=out)