import FileIO
import config

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import cv2
import argparse
import os

def plot_frame_rate_graph(videos_location, data_csv_file):
	videos_list = FileIO.load_csv_as_list(data_csv_file)

	frame_rates = defaultdict(int)

	for video_info in videos_list:
		video_filename = FileIO.get_video_filename(video_info[0], video_info[1], video_info[2], config.video_file_extension)
		video_path = os.path.join(videos_location, video_filename)

		video = cv2.VideoCapture(video_path)

		fps = video.get(cv2.CAP_PROP_FPS)
		frame_rates[fps] += 1

	plt.bar(range(len(frame_rates)), frame_rates.values(), align="center")
	plt.xticks(range(len(frame_rates)), list(frame_rates.keys()))
	plt.savefig('foo.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_recognize = subparsers.add_parser('test')
    parser_recognize = subparsers.add_parser('evaluate')

    args = parser.parse_args()

    if args.mode == 'train':
        plot_frame_rate_graph(config.video_training_data_location, config.training_data_csv_file)
    elif args.mode == 'test':
        plot_frame_rate_graph(config.video_testing_data_location, config.testing_data_csv_file)
    elif args.mode == 'evaluate':
        plot_frame_rate_graph(config.video_evaluation_data_location, config.evaluation_data_csv_file)
