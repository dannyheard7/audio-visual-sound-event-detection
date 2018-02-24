import os
import numpy as np
import argparse
import numpy as np 
import pickle
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

import config
import meta
import FileIO


def take_frame_from_start(video_info, video_path, output_folder):
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

    filenames = []

    for i in range(0, num_frames - 1):
        frame_filename.append(frame_filename % i)

    return filenames


def take_frame_each_scene(video_info, video_filename, video_location, output_folder):
    pass


def extract_image_features(model, image_path, features_output_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print(np.shape(preds))
    # decode the results into a list of tuples (class, description, probability)
    # print('Predicted:', decode_predictions(preds, top=3)[0])

    # Do i need to do PCA?
    pickle.dump(preds, open(features_output_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def get_video_frames(videos_location, output_folder, data_csv_file):
    FileIO.create_folder(output_folder)
    
    features_output_path = os.path.join(output_folder, "features/")
    FileIO.create_folder(features_output_path)

    videos_info = FileIO.load_csv_as_list(data_csv_file)
    model = InceptionV3(weights='imagenet')

    for video_info in videos_info:
        video_filename = FileIO.get_video_filename(video_info[0], video_info[1], video_info[2], config.video_file_extension)
        video_path = os.path.join(videos_location, video_filename)
        frame_features_path = os.path.join(features_output_path, os.path.splitext(video_filename)[0]) + ".pkl"

        if os.path.exists(video_path) and not os.path.exists(frame_features_path):
            frame_filename = take_frame_from_start(video_info, video_path, output_folder)

            if os.path.exists(frame_filename):
                extract_image_features(model, frame_filename, frame_features_path)
                os.remove(frame_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_recognize = subparsers.add_parser('test')
    parser_recognize = subparsers.add_parser('evaluate')

    args = parser.parse_args()

    if args.mode == 'train':
        get_video_frames(config.video_training_data_location, config.video_training_frames_location, config.training_data_csv_file)
    elif args.mode == 'test':
        get_video_frames(config.video_testing_data_location, config.video_testing_frames_location, config.testing_data_csv_file)
    elif args.mode == 'evaluate':
        get_video_frames(config.video_evaluation_data_location, config.video_evaluation_frames_location, config.evaluation_data_csv_file)
