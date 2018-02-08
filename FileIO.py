import csv
import os

def get_video_filename(video_id, start_time, end_time, video_extension):
	return "Y{}_{}_{}{}".format(video_id, start_time, end_time, video_extension) # Download video script places a Y at the start of the filename


def get_frame_filename(video_path, frame_type):
	filename, _ = os.path.splitext(os.path.basename(video_path))
	filename += "_{}{}".format(frame_type, config.video_frames_extension)
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