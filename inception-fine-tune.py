import argparse
import os

from keras import Model
from keras.applications import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, K, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import keras.losses
from sklearn.utils import compute_class_weight
import numpy as np

import config
import meta
from audio_system.evaluation import io_task4


def get_train_labels_list():
    videos_by_class = meta.load_videos_info_by_class(config.training_data_csv_file)
    keys = sorted(list(videos_by_class.keys()), key=str.lower)
    labels = []

    for label in keys:
        labels.extend([label] * len(videos_by_class[label]))

    return labels


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def fine_tune_inception(frames_folder, model_dir):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    top_layers_checkpoint_path = os.path.join(model_dir, 'cp.top.best.hdf5')
    fine_tuned_checkpoint_path = os.path.join(model_dir, 'cp.fine_tuned.best.hdf5')
    new_extended_inception_weights = os.path.join(model_dir, 'final_weights.hdf5')
    
    train_data_dir = os.path.join(frames_folder, 'training')
    validation_data_dir = os.path.join(frames_folder, 'testing')
    img_width, img_height = 299, 299
    nb_train_samples = 41497 
    nb_validation_samples = 396

    labels = get_train_labels_list()
    class_weights = compute_class_weight('balanced', np.unique(labels), labels)
    wcc = weighted_categorical_crossentropy(class_weights)

    top_epochs = 10
    fit_epochs = 65
    batch_size = 50

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
#    x = Dropout(0.8)(x)
    x = Dense(1024, activation='relu')(x) # let's add a fully-connected layer
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)
    predictions = Dense(17, activation='softmax')(x) # and a logistic layer -- we have 17 classes

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    if not os.path.exists(top_layers_checkpoint_path):
        # first: train only the top layers (which were randomly initialized) i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        #model = multi_gpu_model(model, gpus=3)

        model.compile(optimizer='rmsprop', loss=wcc, metrics=['accuracy'])

        mc_top = ModelCheckpoint(top_layers_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

        model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=top_epochs,
                            validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                            callbacks=[mc_top])

    model.load_weights(top_layers_checkpoint_path)        
    print ("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")

    if os.path.exists(fine_tuned_checkpoint_path):
        model.load_weights(fine_tuned_checkpoint_path)
        print("Checkpoint '" + fine_tuned_checkpoint_path + "' loaded.")

    # we chose to train the top 2 inception blocks, i.e. we will freeze the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect. we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss=wcc, metrics=['accuracy'])

    # Save the model after every epoch.
    mc_fit = ModelCheckpoint(fine_tuned_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

    # we train our model again (this time fine-tuning the top 2 inception blocks alongside the top Dense layers
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=fit_epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[mc_fit])

    model.save_weights(new_extended_inception_weights)


def recognize(args):
    frame_dir = args.frame_dir
    model_path = args.model_path
    
    labels = get_train_labels_list()
    class_weights = compute_class_weight('balanced', np.unique(labels), labels)
    wcc = weighted_categorical_crossentropy(class_weights)
    
    model = load_model(model_path, custom_objects={'loss': wcc}) # Audio tagging

    na_list = []
    fusion_at_list = [] 

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        frame_dir,
        target_size=(299, 299),
        color_mode="rgb",
        shuffle = "false",
        class_mode=None,
        batch_size=1)   
   
    labels_indices = [sorted(config.labels, key=str.lower).index(label) for label in config.labels]
    fusion_at = model.predict_generator(test_generator, steps=len(test_generator.filenames))[:, labels_indices] # What order do the predictions need to be?

    for frame_path in test_generator.filenames:
        na, _ = os.path.splitext(os.path.basename(frame_path))
        na = na[1:na.rfind("_")] + ".wav"
        na_list.append(na)

    # Write out AT probabilities
    #fusion_at = np.mean(np.array(fusion_at_list), axis=0)
    print("AT shape: %s" % (fusion_at.shape,))
    na_list = [x.encode('utf-8') for x in na_list]
    
    io_task4.at_write_prob_mat_to_csv(
        na_list=na_list, 
        prob_mat=fusion_at, 
        out_path=os.path.join(args.out_dir, "at_prob_mat.csv.gz"))
            
    print("Prediction finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_finetune = subparsers.add_parser('fine-tune')
    parser_finetune.add_argument('--frames_folder', type=str)
    parser_finetune.add_argument('--model_dir', type=str)

    parser_recognize = subparsers.add_parser('recognize')
    parser_recognize.add_argument('--frame_dir', type=str)
    parser_recognize.add_argument('--model_path', type=str)
    parser_recognize.add_argument('--out_dir', type=str)

    args = parser.parse_args()

    if args.mode == 'fine-tune':
        fine_tune_inception(args.frames_folder, args.model_dir)
    elif args.mode == 'recognize':
        recognize(args)
