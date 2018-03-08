import argparse
import os

import dill as pickle
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image

import FileIO
import config


def take_frame_from_start(video_path, output_folder):
    frame_filename = os.path.join(output_folder, FileIO.get_frame_filename(video_path, "start", config.video_frames_extension))

    ffmpeg_string = "ffmpeg -i {}  -ss 00:00:00.1 -vframes 1 {}".format(video_path, frame_filename)
    os.system(ffmpeg_string)

    return frame_filename


def take_frame_from_middle(video_info, video_path, output_folder):
    frame_filename = os.path.join(output_folder, FileIO.get_frame_filename(video_path, "middle",  config.video_frames_extension))

    middle_location = str((float(video_info[2]) - float(video_info[1])) / 2)
    
    ffmpeg_string = "ffmpeg -i {}  -ss 00:00:0{} -vframes 1 {}".format(video_path, middle_location, frame_filename)
    os.system(ffmpeg_string)

    return frame_filename


def take_frame_from_end(video_info, video_path, output_folder):
    frame_filename = os.path.join(output_folder, FileIO.get_frame_filename(video_path, "end",  config.video_frames_extension))

    end_location = str((float(video_info[2]) - float(video_info[1])))

    if not end_location.startswith("10."):
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

    filenames = []

    for i in range(1, num_frames + 2):
        filenames.append(frame_filename % i)

    return filenames


def take_frame_each_scene(video_info, video_filename, video_location, output_folder):
    pass


def extract_image_features(model, image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds
   

def get_video_frame_features(videos_location, output_folder, data_csv_file):
    FileIO.create_folder(output_folder)
    
    features_output_path = os.path.join(output_folder, "features-pca/")
    FileIO.create_folder(features_output_path)

    videos_info = FileIO.load_csv_as_list(data_csv_file)
    model = InceptionV3(weights='imagenet')

    predictions_video_info = []
    all_predictions = []

    for video_info in videos_info:
        video_filename = FileIO.get_video_filename(video_info[0], video_info[1], video_info[2], config.video_file_extension)
        video_path = os.path.join(videos_location, video_filename)
        frame_features_path = os.path.join(features_output_path, os.path.splitext(video_filename)[0]) + ".pkl"

        if os.path.exists(video_path) and not os.path.exists(frame_features_path):
            frame_filename = take_frame_from_start(video_path, output_folder)

            if os.path.exists(frame_filename):
                predictions = extract_image_features(model, frame_filename)
                os.remove(frame_filename)
                predictions_video_info.append(video_info)
                all_predictions.append(predictions)

    all_predictions = dimensionality_reduction(np.asarray(all_predictions)[:, 0, :], 30) # Why does this turn into a 3-dimensional array?

    for i in range(0, len(predictions_video_info)):
        video_info = predictions_video_info[i]

        video_filename = FileIO.get_video_filename(video_info[0], video_info[1], video_info[2],
                                                   config.video_file_extension)
        frame_features_path = os.path.join(features_output_path, os.path.splitext(video_filename)[0]) + ".pkl"

        pickle.dump(all_predictions[i], open(frame_features_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def dimensionality_reduction(data, num_dims_to_keep):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    cov = np.cov(centered_data.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov)
    p = eigenvalues.argsort()[::-1]  # Sort eigenvalues in descending order
    eigenvectors = eigenvectors[:, p]  # Sort eigenvectors by corresponding eignenvalue

    principal_components = eigenvectors[:, :num_dims_to_keep]
    reduced_data = data.dot(principal_components)

    return reduced_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_get_frames = subparsers.add_parser("get_frames")
    parser_get_frames.add_argument('--videos_location', type=str)
    parser_get_frames.add_argument('--frames_location', type=str)
    parser_get_frames.add_argument('--csv_file', type=str)

    args = parser.parse_args()

    if args.mode == 'get_frames':
        get_video_frame_features(args.videos_location, args.frames_location, args.csv_file)
