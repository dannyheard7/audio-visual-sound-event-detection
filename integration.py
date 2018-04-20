import argparse
import os

import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import load_model
from sklearn.utils import compute_class_weight

import config
import meta
from FileIO import create_folder
from audio_system.data_generator import RatioDataGenerator
from audio_system.evaluation import io_task4
from audio_system.evaluation.io_task4 import at_read_prob_mat_csv


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


def get_stat(audio_only_matrix_path, visual_only_matrix_path, combined_matrix_output_path, submission_csv_output_path):
    create_folder(os.path.dirname(combined_matrix_output_path))
    create_folder(os.path.dirname(submission_csv_output_path))

    labels = config.labels
    threshold_array = [0.25] * len(labels)

    audio_predictions_file_list, audio_predictions_probability_matrix = at_read_prob_mat_csv(
        audio_only_matrix_path)

    visual_predictions_file_list, visual_predictions_probability_matrix = at_read_prob_mat_csv(
        visual_only_matrix_path)

    na_list, audio_predictions_probability_matrix, visual_predictions_probability_matrix = reorder_matrices(
        audio_predictions_file_list, audio_predictions_probability_matrix, visual_predictions_file_list,
        visual_predictions_probability_matrix )

    # Merge predicitions by adding logs of probabilities
    #combined_predictions_probability_matrix = np.exp(np.log(visual_predictions_probability_matrix) + np.log(audio_predictions_probability_matrix))

    #combined_predictions_probability_matrix = (0.10 * visual_predictions_probability_matrix + 0.9 * audio_predictions_probability_matrix) / 2
    combined_predictions_probability_matrix = np.maximum(visual_predictions_probability_matrix, audio_predictions_probability_matrix)
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


def integration_model(input_shape):
    model = Sequential()
    model.add(Dropout(0.5, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dense(34, activation='tanh'))
    #model.add(BatchNormalization(input_shape=input_shape))
    #model.add(Dropout(0.5))
    #model.add(Dense(17,  activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_integration_layer(audio_train_outputs, audio_test_ouputs, visual_train_outputs, visual_test_outputs, model_path):
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

    model = integration_model(input_shape=train_predictions_matrix.shape[1:])

    gen = RatioDataGenerator(batch_size=batch_size, type='train')

    model.fit_generator(generator=gen.generate({'x': train_predictions_matrix, 'y': train_labels}),
                        steps_per_epoch=5.5 * 100,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[mc_top],
                        validation_data=(test_predictions_matrix, test_labels),
                        class_weight=class_weights)


def recognise_integration(audio_eval_outputs, visual_eval_outputs, out_dir, model_path):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_probabilities = subparsers.add_parser('combine_probabilities')
    parser_probabilities.add_argument('--audio_only_matrix', type=str)
    parser_probabilities.add_argument('--visual_only_matrix', type=str)
    parser_probabilities.add_argument('--combined_matrix_output', type=str)
    parser_probabilities.add_argument('--submission_csv_output', type=str)

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--audio_only_train_matrix', type=str)
    parser_train.add_argument('--audio_only_test_matrix', type=str)
    parser_train.add_argument('--visual_only_train_matrix', type=str)
    parser_train.add_argument('--visual_only_test_matrix', type=str)
    parser_train.add_argument('--model_path', type=str)

    parser_recognise = subparsers.add_parser('recognise')
    parser_recognise.add_argument('--audio_only_eval_matrix', type=str)
    parser_recognise.add_argument('--visual_only_eval_matrix', type=str)
    parser_recognise.add_argument('--out_dir', type=str)
    parser_recognise.add_argument('--model_path', type=str)

    args = parser.parse_args()

    if args.mode == 'combine_probabilities':
        get_stat(args.audio_only_matrix, args.visual_only_matrix, args.combined_matrix_output, args.submission_csv_output)
    elif args.mode == 'train':
        train_integration_layer(args.audio_only_train_matrix, args.audio_only_test_matrix, args.visual_only_train_matrix,
                                args.visual_only_test_matrix, args.model_path)
    elif args.mode == 'recognise':
        recognise_integration(args.audio_only_eval_matrix, args.visual_only_eval_matrix, args.out_dir, args.model_path)
