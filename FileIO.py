import argparse
import csv
import os
import itertools

import config
import meta


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


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


def write_list_to_csv(filepath, list_to_write):
    with open(filepath,'wt') as csvfile:
            writer = csv.writer(csvfile)

            for item in list_to_write:
                writer.writerow(item)


def save_data_as_pickle():
    pass


def extract_evaluation_video_info():
    with open('metadata/evaluation_set.csv', 'w') as out, open('metadata/groundtruth_weak_label_evaluation_set.csv', 'r') as in_file:
        for row in csv.reader(in_file, delimiter='\t'):
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

            classes = "\"{}\"".format(row[-1])
            
            print(video_id + ',' + start_time + ',' + end_time + ',' + classes, file=out)


def move_video_frames(videos_location, output_folder, data_csv_file):
    class_labels = meta.load_sound_event_classes()
    videos_by_classes = meta.load_videos_info_by_class(data_csv_file)

    for class_label, class_name in class_labels.items():
        for video_info in videos_by_classes[class_label]:
            video_filename = get_video_filename(video_info[0], video_info[1], video_info[2], config.video_file_extension)
            video_path = os.path.join(videos_location, video_filename)

            frame_filename = get_frame_filename(video_path, "middle", config.video_frames_extension)
            frame_location = os.path.join(output_folder, frame_filename)

            new_frame_folder = os.path.join(output_folder, class_name)

            if not os.path.exists(new_frame_folder):
                os.makedirs(new_frame_folder)

            if os.path.exists(frame_location):
                new_frame_location = os.path.join(new_frame_folder, frame_filename)
                os.rename(frame_location, new_frame_location)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_get_frames = subparsers.add_parser("move_files")
    parser_get_frames.add_argument('--videos_location', type=str)
    parser_get_frames.add_argument('--frames_location', type=str)
    parser_get_frames.add_argument('--csv_file', type=str)

    args = parser.parse_args()

    if args.mode == 'move_files':
        move_video_frames(args.videos_location, args.frames_location, args.csv_file)






