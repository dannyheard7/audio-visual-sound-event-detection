import argparse
import collections
import dill as pickle
import matplotlib as mpl
import os
mpl.use('Agg')
import matplotlib.pyplot as plt
import math


import FileIO
import config
import meta


def plot_preds_by_class(model_path, videos_location, output_folder, data_csv_file):
    import video_frames
    from keras.models import load_model
    from keras.applications.imagenet_utils import decode_predictions

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
                    print(frame_filename)
                    if os.path.exists(frame_filename):
                        preds = video_frames.extract_image_features(model, frame_filename)
                        os.remove(frame_filename)

                        decoded = decode_predictions(preds, top=3)[0]
                        for prediction in decoded:
                            class_predictions[class_name][prediction[1]] += 1  # , prediction[2]))

    class_predictions = {class_name: {prediction_name: int(math.ceil(num_predictions)) / 3 for prediction_name, num_predictions in predictions.items()} 
                    for class_name, predictions in class_predictions.items()}

    print(class_predictions)
    pickle.dump(class_predictions, open("predictions.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def plot_on_graph(predictions_file, output_file):
    class_predictions = pickle.load(open(predictions_file, "rb" ))

    num_classes = len(class_predictions)
    fig, axarr = plt.subplots(num_classes, figsize=(65, 60))

    predictions = {class_name: {prediction_name: num_predictions for prediction_name, num_predictions in predictions.items() if num_predictions > 4} 
                    for class_name, predictions in class_predictions.items()}

    dcase_labels = list(predictions.keys())
    count = 0
    for i in range(0, num_classes):
        if len(dcase_labels) == count:
            break

        class_name = dcase_labels[count]
        class_predictions = predictions[class_name]
        dict_len = len(class_predictions)

        axarr[i].bar(range(dict_len), list(class_predictions.values()), align='center')
        axarr[i].set_title("ImageNet Predicitions for {}".format(class_name))
        axarr[i].set_xticks(range(dict_len))
        axarr[i].set_xticklabels(list(class_predictions.keys()), rotation='vertical')
        count += 1

    fig.tight_layout()
    plt.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('--model_path', type=str)

    subparsers.add_parser('plot-predictions')
    args = parser.parse_args()

    if args.mode == 'predict':
        plot_preds_by_class(args.model_path, config.video_training_data_location, config.video_training_frames_location,
                            config.training_data_csv_file)
    elif args.mode == 'plot-predictions':
        plot_on_graph("predictions.pkl", "predictions-bar-chart3.png")
