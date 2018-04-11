import argparse
import os

import numpy as np

import config
from FileIO import create_folder
from audio_system.evaluation import io_task4
from audio_system.evaluation.io_task4 import at_read_prob_mat_csv


def reorder_matrices(file_list1, prob_mat1, file_list2, prob_mat2):
    list1_indices = [file_list1.index(e) for e in file_list2
                      if e in file_list1]
    prob_mat1 = prob_mat1[list1_indices]

    list2_indices = [file_list2.index(e) for e in file_list1
                     if e in file_list2]
    prob_mat2 = prob_mat2[list2_indices]

    combined_file_list = [file_list1[i].encode('utf-8') for i in list1_indices]

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
    combined_predictions_probability_matrix = np.exp(np.log(visual_predictions_probability_matrix) +
                                                     np.log(audio_predictions_probability_matrix))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_probabilities = subparsers.add_parser('combine_probabilities')
    parser_probabilities.add_argument('--audio_only_matrix', type=str)
    parser_probabilities.add_argument('--visual_only_matrix', type=str)
    parser_probabilities.add_argument('--combined_matrix_output', type=str)
    parser_probabilities.add_argument('--submission_csv_output', type=str)

    args = parser.parse_args()

    if args.mode == 'combine_probabilities':
        get_stat(args.audio_only_matrix, args.visual_only_matrix, args.combined_matrix_output, args.submission_csv_output)

