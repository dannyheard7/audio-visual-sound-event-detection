import numpy as np
import sys
import soundfile
import os
import librosa
from scipy import signal
import pickle
import scipy
import time
import csv
import gzip
import h5py
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
import argparse

import config as cfg

# Read wav
def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
# Write wav
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

# Create an empty folder
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

### Feature extraction. 
def extract_features(wav_dir, out_dir, recompute):
    """Extract log mel spectrogram features. 
    
    Args:
      wav_dir: string, directory of wavs. 
      out_dir: string, directory to write out features. 
      recompute: bool, if True recompute all features, if False skip existed
                 extracted features. 
                 
    Returns:
      None
    """
    fs = cfg.sample_rate
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    
    create_folder(out_dir)
    names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
    names = sorted(names)
    print("Total file number: %d" % len(names))

    # Mel filter bank
    melW = librosa.filters.mel(sr=fs, 
                               n_fft=n_window, 
                               n_mels=64, 
                               fmin=0., 
                               fmax=8000.)
    
    cnt = 0
    t1 = time.time()
    for na in names:
        wav_path = wav_dir + '/' + na
        out_path = out_dir + '/' + os.path.splitext(na)[0] + '.pkl'
        
        # Skip features already computed
        if recompute or (not os.path.isfile(out_path)):
            print(cnt, out_path)
            (audio, _) = read_audio(wav_path, fs)
            
            # Skip corrupted wavs
            if audio.shape[0] == 0:
                print("File %s is corrupted!" % wav_path)
            else:
                # Compute spectrogram
                ham_win = np.hamming(n_window)
                [f, t, x] = signal.spectral.spectrogram(
                                x=audio, 
                                window=ham_win,
                                nperseg=n_window, 
                                noverlap=n_overlap, 
                                detrend=False, 
                                return_onesided=True, 
                                mode='magnitude') 
                x = x.T
                x = np.dot(x, melW.T)
                x = np.log(x + 1e-8)
                x = x.astype(np.float32)

                # Here need to add visual features
                
                # Dump to pickle
                pickle.dump(x, open(out_path, 'wb'), 
                             protocol=pickle.HIGHEST_PROTOCOL)
        cnt += 1
    print("Extracting feature time: %s" % (time.time() - t1,))

### Pack features of hdf5 file
def pack_features_to_hdf5(audio_feature_dir, video_feature_dir, csv_path, out_path):
    """Pack extracted features to a single hdf5 file. 
    
    This hdf5 file can speed up loading the features. This hdf5 file has 
    structure:
       na_list: list of names
       x: bool array, (n_clips)
       y: float32 array, (n_clips, n_time, n_freq)
       
    Args: 
      audio_feature_dir: string, directory of features.
      csv_path: string | "", path of csv file. E.g. "testing_set.csv". If the 
          string is empty, then pack features with all labels False. 
      out_path: string, path to write out the created hdf5 file. 
      
    Returns:
      None
    """
    max_len = cfg.max_len
    create_folder(os.path.dirname(out_path))
    
    t1 = time.time()
    x_all, y_all, na_all = [], [], []

    with h5py.File(out_path, 'w') as hf:  
        x_dset = hf.create_dataset('x', (1, 240, 96), maxshape=(None, 240, 96), dtype='f', chunks=(1, 240, 96))
    
        if csv_path != "":    # Pack from csv file (training & testing from dev. data)         
            with open(csv_path, 'rt') as f:
                reader = csv.reader(f)
                lis = list(reader)

            count = 0
            for li in lis:
                [id, start, end, labels, label_ids] = li
                if count % 100 == 0: print(count)
                # id = os.path.splitext(id)[0]
                filename = 'Y' + id + '_' + start + '_' + end # Correspond to the wav name.
                feature_filename = filename + ".pkl"
                feature_path = os.path.join(audio_feature_dir, feature_filename)

                video_feature_path = os.path.join(video_feature_dir, feature_filename)
                
                if not os.path.isfile(feature_path) or not os.path.isfile(video_feature_path):
                    print("File %s is in the csv file but the feature is not extracted!" % filename)
                else:
                    na_all.append(filename[1:] + ".wav") # Remove 'Y' in the begining.

                    x_audio = pickle.load(open(feature_path, 'rb'))
                    x_audio = pad_trunc_seq(x_audio, max_len)

                    x_video = pickle.load(open(video_feature_path, 'rb')).repeat(240, 0) # Height needs to be 240 like frequency
                    x = np.hstack((x_audio, x_video))
                    
                    x_dset[-1] = x.astype(np.float32)

                    if count != (len(lis) - 1):
                        x_dset.resize(x_dset.shape[0] + 1, axis=0)

                    label_ids = label_ids.split(',')                    
                    y = ids_to_multinomial(label_ids)
                    y_all.append(y)
                count += 1
        else:   # Pack from features without ground truth label (dev. data)
            names = os.listdir(audio_feature_dir)
            names = sorted(names)
            count = 0

            for feature_filename in names:
                filename = os.path.splitext(feature_filename)[0]
                feature_path = os.path.join(audio_feature_dir, feature_filename)
                video_feature_path = os.path.join(video_feature_dir, feature_filename)

                if not os.path.isfile(feature_path) or not os.path.isfile(video_feature_path):
                    print("File %s is in the csv file but the feature is not extracted!" % filename)
                else:
                    na_all.append(filename[1:] + ".wav")
                    x_audio = pickle.load(open(feature_path, 'rb'))
                    x_audio = pad_trunc_seq(x_audio, max_len)

                    x_video = pickle.load(open(video_feature_path, 'rb')).repeat(240, 0) # Height needs to be 240 like frequency
                    print("video shape {}".format(x_video.shape))
                    x = np.hstack((x_audio, x_video))
                    
                    x_dset[-1] = x.astype(np.float32)

                    if count != (len(names) - 1):
                        x_dset.resize(x_dset.shape[0] + 1, axis=0)
                        
                    y_all.append(None)
                    count += 1

        y_all = np.array(y_all, dtype=np.bool)
        hf.create_dataset('y', data=y_all)

        na_all = [x.encode('utf-8') for x in na_all] # convert to utf-8 to store
        hf.create_dataset('na_list', data=na_all)
        
    
    print("Pack features time: %s" % (time.time() - t1,))
    
def ids_to_multinomial(ids):
    """Ids of wav to multinomial representation. 
    
    Args:
      ids: list of id, e.g. ['/m/0284vy3', '/m/02mfyn']
      
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    y = np.zeros(len(cfg.lbs))
    for id in ids:
        index = cfg.id_to_idx[id]
        y[index] = 1
    return y
    
def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length. 
    
    Args:
      x: ndarray, input sequence data. 
      max_len: integer, length of sequence to be padded or truncated. 
      
    Returns:
      ndarray, Padded or truncated input sequence data. 
    """
    L = len(x)
    shape = x.shape
    if L < max_len:
        pad_shape = (max_len - L,) + shape[1:]
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad), axis=0)
    else:
        x_new = x[0:max_len]
    return x_new
    
### Load data & scale data
def load_hdf5_data(hdf5_path, verbose=1):
    """Load hdf5 data. 
    
    Args:
      hdf5_path: string, path of hdf5 file. 
      verbose: integar, print flag. 
      
    Returns:
      x: ndarray (np.float32), shape: (n_clips, n_time, n_freq)
      y: ndarray (np.bool), shape: (n_clips, n_classes)
      na_list: list, containing wav names. 
    """
    t1 = time.time()
    with h5py.File(hdf5_path, 'r') as hf:
        x = np.array(hf.get('x'))
        y = np.array(hf.get('y'))
        na_list = list(hf.get('na_list'))
        
    if verbose == 1:
        print("--- %s ---" % hdf5_path)
        print("x.shape: %s %s" % (x.shape, x.dtype))
        print("y.shape: %s %s" % (y.shape, y.dtype))
        print("len(na_list): %d" % len(na_list))
        print("Loading time: %s" % (time.time() - t1,))
        
    return x, y, na_list

def calculate_scaler(hdf5_train_path, hdf5_test_path, hdf5_eval_path, out_path):
    """Calculate scaler of input data on each frequency bin. 
    
    Args:
      hdf5_path: string, path of packed hdf5 features file. 
      out_path: string, path to write out the calculated scaler. 
      
    Returns:
      None. 
    """
    create_folder(os.path.dirname(out_path))
    t1 = time.time()
    tr_data = h5py.File(hdf5_train_path, 'r+')
    
    count = 0
    batch_size = 10000
    x_audio = []
    for i in range(batch_size, tr_data['x'].shape[0] + batch_size, batch_size):
        if i >= tr_data['x'].shape[0]:
            i = tr_data['x'].shape[0] -1
        x_audio.extend(tr_data['x'][count:i, :, :64])
        print(len(x_audio))
        count += batch_size
    x_audio = np.asarray(x_audio)

    (n_clips, n_time, n_freq) = x_audio.shape
    x2d = x_audio.reshape((n_clips * n_time, n_freq))
    scaler = preprocessing.StandardScaler().fit(x2d)

    print("Calculating scaler time: %s" % (time.time() - t1,))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'wb') as f:
        pickle.dump(scaler, f)

    count = 0
    for i in range(batch_size, tr_data['x'].shape[0] + batch_size, batch_size):
        if i >= tr_data['x'].shape[0]:
            i = tr_data['x'].shape[0] -1

        tr_data['x'][count:i, :, :64] = do_scale(tr_data['x'][count:i, :, :64], scaler, verbose=1)
        count += batch_size

    te_data = h5py.File(hdf5_test_path, 'r+')
    count = 0
    for i in range(batch_size, te_data['x'].shape[0] + batch_size, batch_size):
        if i >= te_data['x'].shape[0]:
            i = te_data['x'].shape[0] -1

        te_data['x'][count:i, :, :64] = do_scale(te_data['x'][count:i, :, :64], scaler, verbose=1)
        count += batch_size

    ev_data = h5py.File(hdf5_eval_path, 'r+')
    count = 0
    for i in range(batch_size, ev_data['x'].shape[0] + batch_size, batch_size):
        if i >= ev_data['x'].shape[0]:
            i = ev_data['x'].shape[0] -1

        ev_data['x'][count:i, :, :64] = do_scale(ev_data['x'][count:i, :, :64], scaler, verbose=1)
        count += batch_size
    
def do_scale(x3d, scaler, verbose=1):
    """Do scale on the input sequence data. 
    
    Args:
      x3d: ndarray, input sequence data, shape: (n_clips, n_time, n_freq)
      scaler: pre-computed scalar 
      verbose: integar, print flag. 
      
    Returns:
      Scaled input sequence data. 
    """
    t1 = time.time()
    (n_clips, n_time, n_freq) = x3d.shape
    x2d = x3d.reshape((n_clips * n_time, n_freq))
    x2d_scaled = scaler.transform(x2d)
    x3d_scaled = x2d_scaled.reshape((n_clips, n_time, n_freq))
    if verbose == 1:
        print("Scaling time: %s" % (time.time() - t1,))
    return x3d_scaled

### Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_ef = subparsers.add_parser('extract_features')
    parser_ef.add_argument('--wav_dir', type=str)
    parser_ef.add_argument('--out_dir', type=str)
    parser_ef.add_argument('--recompute', type=bool)
    
    parser_pf = subparsers.add_parser('pack_features')
    parser_pf.add_argument('--audio_feature_dir', type=str)
    parser_pf.add_argument('--video_feature_dir', type=str)
    parser_pf.add_argument('--csv_path', type=str)
    parser_pf.add_argument('--out_path', type=str)
    
    parser_cs = subparsers.add_parser('calculate_scaler')
    parser_cs.add_argument('--hdf5_train_path', type=str)
    parser_cs.add_argument('--hdf5_test_path', type=str)
    parser_cs.add_argument('--hdf5_eval_path', type=str)
    parser_cs.add_argument('--out_path', type=str)

    args = parser.parse_args()
    
    if args.mode == 'extract_features':
        extract_features(wav_dir=args.wav_dir, 
                         out_dir=args.out_dir, 
                         recompute=args.recompute)
    elif args.mode == 'pack_features':
        pack_features_to_hdf5(audio_feature_dir=args.audio_feature_dir,
                              video_feature_dir=args.video_feature_dir,
                              csv_path=args.csv_path,
                              out_path=args.out_path)
    elif args.mode == 'calculate_scaler':
        calculate_scaler(hdf5_train_path=args.hdf5_train_path, 
                        hdf5_test_path=args.hdf5_test_path,
                        hdf5_eval_path=args.hdf5_eval_path,
                         out_path=args.out_path)
    else:
        raise Exception("Incorrect argument!")
