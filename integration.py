import argparse
import csv
import os
import pickle

import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import load_model
from sklearn.utils import compute_class_weight
import h5py

import config
import meta
from FileIO import create_folder
from audio_system.data_generator import RatioDataGenerator
from audio_system.evaluation import io_task4
from audio_system.evaluation.io_task4 import at_read_prob_mat_csv
from audio_system.prepare_data import ids_to_multinomial, load_hdf5_data


def reorder_matrices(file_list1, prob_mat1, file_list2, prob_mat2):
    list1_indices = [file_list1.index(e) for e in file_list2
                      if e in file_list1]
    file_list1 = [file_list1[i] for i in list1_indices]
    prob_mat1 = prob_mat1[list1_indices]

    list2_indices = [file_list2.index(e) for e in file_list1
                     if e in file_list2]
    
    file_list2 = [file_list2[i] for i in list2_indices]
    prob_mat2 = prob_mat2[list2_indices]


    combined_file_list = [x.encode('utf-8') for x in file_list2]
    return combined_file_list, prob_mat1, prob_mat2


def combine_probabilities_linear(audio_only_matrix_path, visual_only_matrix_path, combined_matrix_output_path, submission_csv_output_path):
    create_folder(os.path.dirname(combined_matrix_output_path))
    create_folder(os.path.dirname(submission_csv_output_path))

    labels = config.labels
    threshold_array = [0.30] * len(labels)

    audio_predictions_file_list, audio_predictions_probability_matrix = at_read_prob_mat_csv(
        audio_only_matrix_path)

    visual_predictions_file_list, visual_predictions_probability_matrix = at_read_prob_mat_csv(
        visual_only_matrix_path)

    na_list, audio_predictions_probability_matrix, visual_predictions_probability_matrix = reorder_matrices(
        audio_predictions_file_list, audio_predictions_probability_matrix, visual_predictions_file_list,
        visual_predictions_probability_matrix )

    # Merge predicitions by adding logs of probabilities
    alpha = 0.93
    combined_predictions_probability_matrix = np.exp((1-alpha)*np.log(visual_predictions_probability_matrix) + alpha*np.log(audio_predictions_probability_matrix))
    #combined_predictions_probability_matrix = (visual_predictions_probability_matrix + audio_predictions_probability_matrix) / 2
    #combined_predictions_probability_matrix = np.maximum(visual_predictions_probability_matrix, audio_predictions_probability_matrix)
    shape = combined_predictions_probability_matrix.shape
    combined_predictions_probability_matrix = combined_predictions_probability_matrix.reshape((shape[0], 1, shape[1]))

    # Write combined matrix to csv file
    io_task4.sed_write_prob_mat_list_to_csv(
        na_list=na_list,
        prob_mat_list=combined_predictions_probability_matrix,
        out_path=combined_matrix_output_path)

    # Write AT to submission format
    io_task4.at_write_prob_mat_csv_to_submission_csv(
        at_prob_mat_path=combined_matrix_output_path,
        lbs=labels,
        thres_ary=threshold_array,
        out_path=submission_csv_output_path)


def train_probabilities_integration_layer(audio_train_outputs, audio_test_ouputs, visual_train_outputs, visual_test_outputs, model_path):
    audio_train_predictions_file_list, audio_train_predictions_probability_matrix = at_read_prob_mat_csv(audio_train_outputs)
    audio_test_predictions_file_list, audio_test_predictions_probability_matrix = at_read_prob_mat_csv(audio_test_ouputs)

    visual_train_predictions_file_list, visual_train_predictions_probability_matrix = at_read_prob_mat_csv(visual_train_outputs)
    visual_test_predictions_file_list, visual_test_predictions_probability_matrix = at_read_prob_mat_csv(visual_test_outputs)

    train_na_list, audio_train_predictions_probability_matrix, visual_train_predictions_file_list = reorder_matrices(
        audio_train_predictions_file_list, audio_train_predictions_probability_matrix, visual_train_predictions_file_list,
        visual_train_predictions_probability_matrix)

    test_na_list, audio_test_predictions_probability_matrix, visual_test_predictions_probability_matrix = reorder_matrices(
        visual_test_predictions_file_list, visual_test_predictions_probability_matrix,
        audio_test_predictions_file_list, audio_test_predictions_probability_matrix)

    train_predictions_matrix = np.hstack((audio_train_predictions_probability_matrix, visual_train_predictions_probability_matrix))
    test_predictions_matrix = np.hstack((audio_test_predictions_probability_matrix, visual_test_predictions_probability_matrix))

    # Load in  labels
    train_labels = meta.get_labels(train_na_list, "metadata/training_set.csv")
    train_labels = np.asarray(train_labels)

    test_labels = meta.get_labels(test_na_list, "metadata/testing_set.csv")
    test_labels = np.asarray(test_labels)
    
    labels = meta.get_train_labels_list()
    class_weights = compute_class_weight('balanced', np.unique(labels), labels)

    batch_size = 32
    epochs = 50

    create_folder(os.path.dirname(model_path))
    mc_top = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

    input_shape = train_predictions_matrix.shape[1:]

    model = Sequential([
        # Dropout(0.5, input_shape=input_shape),
        #BatchNormalization(input_shape=input_shape),
        Dense(256, activation='tanh', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='tanh'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='tanh'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='tanh'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(17, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    gen = RatioDataGenerator(batch_size=batch_size, type='train')

    model.fit_generator(generator=gen.generate({'x': train_predictions_matrix, 'y': train_labels}),
                        steps_per_epoch=100,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[mc_top],
                        validation_data=(test_predictions_matrix, test_labels),
                        class_weight=class_weights)


def recognise_probabilities_integration(audio_eval_outputs, visual_eval_outputs, out_dir, model_path):
    audio_predictions_file_list, audio_predictions_probability_matrix = at_read_prob_mat_csv(audio_eval_outputs)

    visual_predictions_file_list, visual_predictions_probability_matrix = at_read_prob_mat_csv(visual_eval_outputs)

    na_list, audio_predictions_probability_matrix, visual_predictions_probability_matrix = reorder_matrices(
        audio_predictions_file_list, audio_predictions_probability_matrix, visual_predictions_file_list,
        visual_predictions_probability_matrix)

    combined_predictions = np.hstack((audio_predictions_probability_matrix, visual_predictions_probability_matrix))

    model = load_model(model_path)  # Audio tagging

    labels_indices = [sorted(config.labels, key=str.lower).index(label) for label in config.labels]
    fusion_at = model.predict(combined_predictions)[:, labels_indices] #, steps=len(combined_predictions))

    create_folder(os.path.dirname(out_dir))
    io_task4.at_write_prob_mat_to_csv(na_list=na_list, prob_mat=fusion_at, out_path=out_dir)


def pack_features(audio_train_outputs, video_feature_dir, csv_path, out_path):
    create_folder(os.path.dirname(out_path))

    audio_predictions_file_list, audio_predictions_probability_matrix = at_read_prob_mat_csv(audio_train_outputs)
    x_all, y_all, na_all = [], [], []

    with h5py.File(out_path, 'w') as hf:
        x_dset = hf.create_dataset('x', (1, 1017), maxshape=(None, 1017), dtype='f', chunks=(1, 1017))
        count = 0

        if csv_path != "":
            with open(csv_path, 'rt') as f:
                reader = csv.reader(f)
                lis = list(reader)

            for li in lis:
                [id, start, end, labels, label_ids] = li
                if count % 100 == 0: print(count)

                filename = 'Y' + id + '_' + start + '_' + end  # Correspond to the wav name.
                feature_filename = filename + ".pkl"
                audio_filename = filename[1:] + ".wav"

                audio_feature_index = audio_predictions_file_list.index(audio_filename) if audio_filename in audio_predictions_file_list else None
                video_feature_path = os.path.join(video_feature_dir, feature_filename)
                
                if audio_feature_index is None or not os.path.isfile(video_feature_path):
                    print("File %s is in the csv file but the feature is not extracted!" % filename)
                else:
                    na_all.append(audio_filename)

                    x_audio = audio_predictions_probability_matrix[audio_feature_index]
                    x_video = pickle.load(open(video_feature_path, 'rb'))
                    x_video = x_video.reshape(x_video.shape[1])
                    x = np.hstack((x_audio, x_video))

                    x_dset[-1] = x.astype(np.float32)

                    if count != (len(lis) - 1):
                        x_dset.resize(x_dset.shape[0] + 1, axis=0)

                    label_ids = label_ids.split(',')
                    y = ids_to_multinomial(label_ids)
                    y_all.append(y)
                count += 1
        else:  # Pack from features without ground truth label (dev. data)
            names = os.listdir(video_feature_dir)
            names = sorted(names)

            for feature_filename in names:
                filename = os.path.splitext(feature_filename)[0]
                audio_filename = filename[1:] + ".wav"

                audio_feature_index = audio_predictions_file_list.index(audio_filename) if audio_filename in audio_predictions_file_list else None
                video_feature_path = os.path.join(video_feature_dir, feature_filename)

                if audio_feature_index is None or not os.path.isfile(video_feature_path):
                    print("File %s is in the csv file but the feature is not extracted!" % filename)
                else:
                    na_all.append(audio_filename)

                    x_audio = audio_predictions_probability_matrix[audio_feature_index]
                    x_video = pickle.load(open(video_feature_path, 'rb'))
                    x_video = x_video.reshape(x_video.shape[1])
                    x = np.hstack((x_audio, x_video))

                    x_dset[-1] = x.astype(np.float32)

                    if count != (len(names) - 1):
                        x_dset.resize(x_dset.shape[0] + 1, axis=0)

                    y_all.append(None)
                    count += 1

        y_all = np.array(y_all, dtype=np.bool)
        hf.create_dataset('y', data=y_all)

        na_all = [x.encode('utf-8') for x in na_all]  # convert to utf-8 to store
        hf.create_dataset('na_list', data=na_all)


def train_features_integration_layer(train_features_path, test_features_path, model_path):
    tr_data = h5py.File(train_features_path, 'r+')
    te_data = h5py.File(test_features_path, 'r+')

    labels = meta.get_train_labels_list()
    class_weights = compute_class_weight('balanced', np.unique(labels), labels)

    batch_size = 64
    epochs = 200

    create_folder(os.path.dirname(model_path))
    mc_top = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

    input_shape = tr_data['x'].shape[1:]

    model = Sequential([
        BatchNormalization(input_shape=input_shape),
        Dropout(0.5),
        #Dense(2048, activation='tanh'),
        #BatchNormalization(),
        #Dropout(0.5),
        Dense(1024, activation='tanh'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='tanh'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='tanh'),
        BatchNormalization(),
        Dropout(0.5),
      #  Dense(256, activation='relu'),
      #  BatchNormalization(),
     #   Dropout(0.6),
        Dense(128, activation='tanh'),
        BatchNormalization(),
        Dropout(0.5),
     #   Dense(32, activation='relu'),
     #   BatchNormalization(),
     #   Dropout(0.6),
        Dense(17, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    gen = RatioDataGenerator(batch_size=batch_size, type='train')

    model.fit_generator(generator=gen.generate(tr_data),
                        steps_per_epoch= 100,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[mc_top],
                        validation_data=(te_data['x'], te_data['y']),
                        class_weight=class_weights)


def recognise_features_integration(eval_features_path, out_dir, model_path):
    (te_x, _, te_na_list) = load_hdf5_data(eval_features_path, verbose=1)

    model = load_model(model_path)  # Audio tagging
    labels_indices = [sorted(config.labels, key=str.lower).index(label) for label in config.labels]
    fusion_at = model.predict(te_x) # [:, labels_indices]

    create_folder(os.path.dirname(out_dir))
    io_task4.at_write_prob_mat_to_csv(na_list=te_na_list, prob_mat=fusion_at, out_path=out_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_probabilities = subparsers.add_parser('combine-probabilities')
    parser_probabilities.add_argument('--audio_only_matrix', type=str)
    parser_probabilities.add_argument('--visual_only_matrix', type=str)
    parser_probabilities.add_argument('--combined_matrix_output', type=str)
    parser_probabilities.add_argument('--submission_csv_output', type=str)

    parser_train = subparsers.add_parser('train-probabilities')
    parser_train.add_argument('--audio_only_train_matrix', type=str)
    parser_train.add_argument('--audio_only_test_matrix', type=str)
    parser_train.add_argument('--visual_only_train_matrix', type=str)
    parser_train.add_argument('--visual_only_test_matrix', type=str)
    parser_train.add_argument('--model_path', type=str)

    parser_recognise = subparsers.add_parser('recognise-probabilities')
    parser_recognise.add_argument('--audio_only_eval_matrix', type=str)
    parser_recognise.add_argument('--visual_only_eval_matrix', type=str)
    parser_recognise.add_argument('--out_dir', type=str)
    parser_recognise.add_argument('--model_path', type=str)

    parser_recognise = subparsers.add_parser('pack-features')
    parser_recognise.add_argument('--audio_train_outputs', type=str)
    parser_recognise.add_argument('--video_feature_dir', type=str)
    parser_recognise.add_argument('--csv_path', type=str)
    parser_recognise.add_argument('--out_path', type=str)

    parser_recognise = subparsers.add_parser('train-features')
    parser_recognise.add_argument('--train_features_path', type=str)
    parser_recognise.add_argument('--test_features_path', type=str)
    parser_recognise.add_argument('--model_path', type=str)

    parser_recognise = subparsers.add_parser('recognise-features')
    parser_recognise.add_argument('--eval_features_path', type=str)
    parser_recognise.add_argument('--out_dir', type=str)
    parser_recognise.add_argument('--model_path', type=str)

    args = parser.parse_args()

    if args.mode == 'combine-probabilities':
        combine_probabilities_linear(args.audio_only_matrix, args.visual_only_matrix, args.combined_matrix_output, args.submission_csv_output)
    elif args.mode == 'train-probabilities':
        train_probabilities_integration_layer(args.audio_only_train_matrix, args.audio_only_test_matrix, args.visual_only_train_matrix,
                                              args.visual_only_test_matrix, args.model_path)
    elif args.mode == 'recognise-probabilities':
        recognise_probabilities_integration(args.audio_only_eval_matrix, args.visual_only_eval_matrix, args.out_dir, args.model_path)
    elif args.mode == 'pack-features':
        pack_features(args.audio_train_outputs, args.video_feature_dir, args.csv_path, args.out_path)
    elif args.mode == 'train-features':
        train_features_integration_layer(args.train_features_path, args.test_features_path, args.model_path)
    elif args.mode == 'recognise-features':
        recognise_features_integration(args.eval_features_path, args.out_dir, args.model_path)
