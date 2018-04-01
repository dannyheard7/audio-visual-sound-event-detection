import argparse
import math
import os

import collections
import dill as pickle

import FileIO
import config
import meta


def save_imagenet_preds_by_class(output_path, videos_location, output_folder, data_csv_file):
    import video_frames
    from keras.applications import InceptionV3
    from keras.applications.imagenet_utils import decode_predictions

    FileIO.create_folder(output_folder)

    class_labels = meta.load_sound_event_classes()
    videos_by_classes = meta.load_videos_info_by_class(data_csv_file)
    model = InceptionV3(weights='imagenet')
    class_predictions = collections.defaultdict(lambda: collections.defaultdict(int))
    num_frames = 3

    for class_label, class_name in class_labels.items():
        for video_info in videos_by_classes[class_label]:
            video_filename = FileIO.get_video_filename(video_info[0], video_info[1], video_info[2],
                                                       config.video_file_extension)
            video_path = os.path.join(videos_location, video_filename)

            if os.path.exists(video_path):
                frame_filenames = video_frames.take_n_equal_spaced_frames(num_frames, video_info, video_path, output_folder)

                for frame_filename in frame_filenames:
                    if os.path.exists(frame_filename):
                        preds = video_frames.extract_image_features(model, frame_filename)
                        os.remove(frame_filename)

                        decoded = decode_predictions(preds, top=3)[0]
                        for prediction in decoded:
                            class_predictions[class_name][prediction[1]] += 1  # , prediction[2]))

    class_predictions = {class_name: {prediction_name: int(math.ceil(num_predictions)) / 3 for prediction_name, num_predictions in predictions.items()} 
                    for class_name, predictions in class_predictions.items()}
    pickle.dump(class_predictions, open(output_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def save_finetuned_preds_by_class(model_path, output_path, videos_location, output_folder, data_csv_file):
    import video_frames
    from keras.models import load_model

    labels = sorted(config.labels, key=str.lower)

    FileIO.create_folder(output_folder)

    class_labels = meta.load_sound_event_classes()
    videos_by_classes = meta.load_videos_info_by_class(data_csv_file)
    model = load_model(model_path)
    class_predictions = collections.defaultdict(lambda: collections.defaultdict(int))
    num_frames = 3

    for class_label, class_name in class_labels.items():
        for video_info in videos_by_classes[class_label]:
            video_filename = FileIO.get_video_filename(video_info[0], video_info[1], video_info[2],
                                                       config.video_file_extension)
            video_path = os.path.join(videos_location, video_filename)

            if os.path.exists(video_path):
                frame_filenames = video_frames.take_n_equal_spaced_frames(num_frames, video_info, video_path, output_folder)

                for frame_filename in frame_filenames:
                    if os.path.exists(frame_filename):
                        preds = video_frames.extract_image_features(model, frame_filename)[0]
                        os.remove(frame_filename)

                        top_predictions = preds.argsort()[-3:][::-1]
                        for prediction_id in top_predictions:
                            class_name = labels[prediction_id]
                            prediction = preds[prediction_id]

                            class_predictions[class_name][prediction] += 1

    class_predictions = {class_name: {prediction_name: int(math.ceil(num_predictions)) / 3 for prediction_name, num_predictions in predictions.items()}
                    for class_name, predictions in class_predictions.items()}

    print(class_predictions)
    pickle.dump(class_predictions, open(output_path, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)


def plot_on_graph(predictions_file):
    import matplotlib.pyplot as plt

    class_predictions = pickle.load(open(predictions_file, "rb" ))

    num_classes = len(class_predictions)
    predictions = {class_name: {prediction_name: num_predictions for prediction_name, num_predictions in predictions.items() if num_predictions > 4} 
                    for class_name, predictions in class_predictions.items()}

    dcase_labels = list(predictions.keys())
    for i in range(0, num_classes):
        fig = plt.figure()

        class_name = dcase_labels[i]
        class_predictions = predictions[class_name]
        dict_len = len(class_predictions)

        plt.bar(range(dict_len), list(class_predictions.values()), align='center')
        plt.title("InceptionV3 Predicitions for {}".format(class_name))
        plt.xticks(range(dict_len), list(class_predictions.keys()), rotation='vertical')
        plt.ylabel('Number of predictions')
        plt.xlabel('Predicted class')

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_imagenet = subparsers.add_parser('predict-imagenet')
    parser_imagenet.add_argument('--output_path', type=str)

    parser_finetuned = subparsers.add_parser('predict-finetuned')
    parser_finetuned.add_argument('--model_path', type=str)
    parser_finetuned.add_argument('--output_path', type=str)

    parser_plot = subparsers.add_parser('plot-predictions')
    parser_plot.add_argument('--predictions_path', type=str)

    args = parser.parse_args()

    if args.mode == 'predict-imagenet':
        save_imagenet_preds_by_class(args.output_path, config.video_training_data_location,
                                      config.video_training_frames_location, config.training_data_csv_file)
    elif args.mode == 'predict-finetuned':
        save_finetuned_preds_by_class(args.model_path, args.output_path, config.video_training_data_location,
                                      config.video_training_frames_location, config.training_data_csv_file)
    elif args.mode == 'plot-predictions':
        plot_on_graph(args.predictions_file)
