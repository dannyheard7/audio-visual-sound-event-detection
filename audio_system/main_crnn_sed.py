"""
Summary:  DCASE 2017 task 4 Large-scale weakly supervised 
          sound event detection for smart cars. Ranked 1 in DCASE 2017 Challenge.
Author:   Yong Xu, Qiuqiang Kong
Created:  03/04/2017
Modified: 31/10/2017
"""
 
import sys
import pickle
import numpy as np
import argparse
import glob
import time
import os
import h5py

import keras
from keras import backend as K
from keras.models import Sequential,Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import Merge, Input, merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
import h5py
from keras.layers.merge import Multiply
from sklearn import preprocessing
import random

import config as cfg
from prepare_data import create_folder, load_hdf5_data, do_scale
from data_generator import RatioDataGenerator
from evaluation import io_task4, evaluate

# CNN with Gated linear unit (GLU) block
def block(input):
    cnn = Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out

def slice1(x):
    return x[:, :, :, 0:64]

def slice2(x):
    return x[:, :, :, 64:128]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

# Attention weighted sum
def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out


def create_model(num_classes, data_shape):
    (_, n_time, n_freq) = data_shape   # (N, 240, 64)
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 64)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 240, 64, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 32, 128)
    print(a1._keras_shape)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 16, 128) # Maybe changing the size of an intermediate layer will help?
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 3))(a1) # (N, 240, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 4, 128) # Should these be 2,2 like the diagram?
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    print(a1._keras_shape)
    a1 = MaxPooling2D(pool_size=(1,4))(a1) # (N, 240, 1, 256)
    
    a1 = Reshape((240, 256))(a1) # (N, 240, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)
    model.summary()
    
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    return model


# Train model
def train(args):
    num_classes = cfg.num_classes

    tr_data = h5py.File(args.tr_hdf5_path, 'r+')
    te_data = h5py.File(args.te_hdf5_path, 'r+')

    tr_shape = tr_data['x'].shape

    print("tr_x.shape: %s" % (tr_shape,))
    
    # Build model
    model = create_model(num_classes, tr_shape)
    
    # Save model callback
    filepath = os.path.join(args.out_model_dir, "gatedAct_rationBal44_lr0.001_normalization_at_cnnRNN_64newMel_240fr.{epoch:02d}-{val_acc:.4f}.hdf5")
    print(filepath)
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc', 
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  
    num_examples = 41498
    batch_size = 8

    # Data generator
    gen = RatioDataGenerator(batch_size=batch_size, type='train')

    # Train
    model.fit_generator(generator=gen.generate(tr_data), 
                        steps_per_epoch=5.5*100,    # 100 iters is called an 'epoch'
                        epochs=31,              # Maximum 'epoch' to train - With larger dataset loss increased after epoch 28
                        verbose=1, 
                        callbacks=[save_model], 
                        validation_data=(te_data['x'], te_data['y']))

# Run function in mini-batch to save memory. 
def run_func(func, x, batch_size):
    pred_all = []
    batch_num = int(np.ceil(len(x) / float(batch_size)))
    for i1 in range(batch_num):
        batch_x = x[batch_size * i1 : batch_size * (i1 + 1)]
        [preds] = func([batch_x, 0.])
        pred_all.append(preds)
    pred_all = np.concatenate(pred_all, axis=0)
    return pred_all

# Recognize and write probabilites. 
def recognize(args, at_bool, sed_bool):
    (te_x, _, te_na_list) = load_hdf5_data(args.hdf5_path, verbose=1)
    x = te_x
    na_list = te_na_list
    
    # x[:, :64] = do_scale(x[:, :64], args.scaler_path, verbose=1)
    
    fusion_at_list = []
    fusion_sed_list = []
    for epoch in range(25, 30, 1):
        t1 = time.time()

        file_name = os.path.join(args.model_dir, "*.%02d-0.*.hdf5" % epoch)
        
        model_path = glob.glob(file_name)[0] # returns more than one item so can't unpack
        model = load_model(model_path)
        
        # Audio tagging
        if at_bool:
            pred = model.predict(x, batch_size=5)
            fusion_at_list.append(pred)
        
        # Sound event detection
        if sed_bool:
            in_layer = model.get_layer('in_layer')
            loc_layer = model.get_layer('localization_layer')
            func = K.function([in_layer.input, K.learning_phase()], 
                              [loc_layer.output])
            pred3d = run_func(func, x, batch_size=20)
            fusion_sed_list.append(pred3d)
        
        print("Prediction time: %s" % (time.time() - t1,))
    
    # Write out AT probabilities
    if at_bool:
        fusion_at = np.mean(np.array(fusion_at_list), axis=0)
        print("AT shape: %s" % (fusion_at.shape,))
        io_task4.at_write_prob_mat_to_csv(
            na_list=na_list, 
            prob_mat=fusion_at, 
            out_path=args.out_dir) # "at_audio_prob_mat.csv.gz"
    
    # Write out SED probabilites
    if sed_bool:
        fusion_sed = np.mean(np.array(fusion_sed_list), axis=0)
        print("SED shape:%s" % (fusion_sed.shape,))
        io_task4.sed_write_prob_mat_list_to_csv(
            na_list=na_list, 
            prob_mat_list=fusion_sed, 
            out_path=os.path.join(args.out_dir, "sed_prob_mat_list.csv.gz"))
            
    print("Prediction finished!")

# Get stats from probabilites. 
def get_stat(args, at_bool, sed_bool):
    labels = cfg.lbs
    step_time_in_sec = cfg.step_time_in_sec
    max_len = cfg.max_len
    threshold_array = [0.35] * len(labels)

    if args.eval:
        weak_gt_csv = "meta_data/groundtruth_weak_label_evaluation_set.csv"
        strong_gt_csv="meta_data/groundtruth_strong_label_evaluation_set.csv"
    else:
        weak_gt_csv="meta_data/groundtruth_weak_label_testing_set.csv"
        strong_gt_csv="meta_data/groundtruth_strong_label_testing_set.csv"

    # Calculate AT stat
    if at_bool:
        prediction_probability_matrix_csv_path = os.path.join(args.pred_dir, "at_audio_prob_mat.csv.gz")
        audio_tagging_stat_path = os.path.join(args.stat_dir, "at_stat.csv")
        audio_tagging_submission_path = os.path.join(args.submission_dir, "at_submission.csv")
        
        audio_tagging_evaluator = evaluate.AudioTaggingEvaluate(
            weak_gt_csv=weak_gt_csv, 
            labels=labels)
        
        at_stat = audio_tagging_evaluator.get_stats_from_probability_matrix_csv(
                        predictions_probability_matrix_csv=prediction_probability_matrix_csv_path,
                        threshold_array=threshold_array)
                        
        # Write out & print AT stat
        audio_tagging_evaluator.write_stat_to_csv(stat=at_stat,
                                       stat_path=audio_tagging_stat_path)
        audio_tagging_evaluator.print_stat(stat_path=audio_tagging_stat_path)
        
        # Write AT to submission format
        io_task4.at_write_prob_mat_csv_to_submission_csv(
            at_prob_mat_path=prediction_probability_matrix_csv_path,
            lbs=labels,
            thres_ary=at_stat['thres_ary'], 
            out_path=audio_tagging_submission_path)
               
    # Calculate SED stat
    if sed_bool:
        sed_prob_mat_list_path = os.path.join(args.pred_dir, "sed_prob_mat_list.csv.gz")
        sed_stat_path = os.path.join(args.stat_dir, "sed_stat.csv")
        sed_submission_path = os.path.join(args.submission_dir, "sed_submission.csv")
        
        sed_evaluator = evaluate.SoundEventDetectionEvaluate(
            strong_gt_csv=strong_gt_csv, 
            lbs=labels,
            step_sec=step_time_in_sec, 
            max_len=max_len)
                            
        # Write out & print SED stat
        sed_stat = sed_evaluator.get_stats_from_prob_mat_list_csv(
                    pd_prob_mat_list_csv=sed_prob_mat_list_path, 
                    thres_ary=threshold_array)
                    
        # Write SED to submission format
        sed_evaluator.write_stat_to_csv(stat=sed_stat, 
                                        stat_path=sed_stat_path)                     
        sed_evaluator.print_stat(stat_path=sed_stat_path)
        
        # Write SED to submission format
        io_task4.sed_write_prob_mat_list_csv_to_submission_csv(
            sed_prob_mat_list_path=sed_prob_mat_list_path, 
            lbs=labels,
            thres_ary=threshold_array,
            step_sec=step_time_in_sec, 
            out_path=sed_submission_path)
                                                        
    print("Calculating stat finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--tr_hdf5_path', type=str)
    parser_train.add_argument('--te_hdf5_path', type=str)
    parser_train.add_argument('--out_model_dir', type=str)
    
    parser_recognize = subparsers.add_parser('recognize')
    parser_recognize.add_argument('--hdf5_path', type=str)
    parser_recognize.add_argument('--model_dir', type=str)
    parser_recognize.add_argument('--out_dir', type=str)
    
    parser_get_stat = subparsers.add_parser('get_stat')
    parser_get_stat.add_argument('--pred_dir', type=str)
    parser_get_stat.add_argument('--stat_dir', type=str)
    parser_get_stat.add_argument('--submission_dir', type=str)
    parser_get_stat.add_argument('--eval', type=bool)
    parser_get_stat.add_argument('--sed', type=bool)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'recognize':
        recognize(args, at_bool=True, sed_bool=False)
    elif args.mode == 'get_stat':
        get_stat(args, at_bool=True, sed_bool=False)
    else:
        raise Exception("Incorrect argument!")
