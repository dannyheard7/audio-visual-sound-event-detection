import argparse
import os
from itertools import cycle, chain, repeat

import numpy as np
from keras import Model
from keras.applications import InceptionV3, VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import compute_class_weight

import config
import meta
from FileIO import create_folder
from audio_system.evaluation import io_task4


def grouper(n, iterable, padvalue=None):
    g = cycle(zip(*[chain(iterable, repeat(padvalue, n - 1))] * n))
    for batch in g:
        yield list(filter(None, batch))


def multilabel_flow(path_to_data, image_datagen, targets, batch_size=256, target_size=(299, 299), sub_folder='training-all'):
    gen = image_datagen.flow_from_directory(path_to_data, batch_size=batch_size, target_size=target_size, classes=[sub_folder],
                                            shuffle=False)
    names_generator = grouper(batch_size, gen.filenames)
    for (X_batch, _), names in zip(gen, names_generator):
        Y_batch = [targets[n.split('/')[-1]] for n in names]
        Y_batch = np.vstack(Y_batch)
        yield X_batch, Y_batch


def fine_tune_inception(frames_folder, model_dir):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    top_layers_checkpoint_path = os.path.join(model_dir, 'cp.top.best.hdf5')
    fine_tuned_checkpoint_path = os.path.join(model_dir, 'cp.fine_tuned.best.hdf5')
    create_folder(model_dir)

    img_width, img_height = 299, 299
    nb_train_samples = 41497 
    nb_validation_samples = 396

    labels = meta.get_train_labels_list()
    class_weights = compute_class_weight('balanced', np.unique(labels), labels)

    top_epochs = 10
    fit_epochs = 65
    batch_size = 64

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

    train_labels = meta.get_images_labels("metadata/training_set.csv")
    test_labels = meta.get_images_labels("metadata/testing_set.csv")

    train_generator = multilabel_flow(frames_folder, train_datagen, train_labels, batch_size=batch_size, target_size=(img_height, img_width),
                                      sub_folder='training-all')

    validation_generator = multilabel_flow(frames_folder, test_datagen, test_labels, batch_size=batch_size, target_size=(img_height, img_width),
                                           sub_folder='testing-all')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)
    x = Dense(1024, activation='relu')(x) # let's add a fully-connected layer
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)
    predictions = Dense(17, activation='sigmoid')(x) # and a logistic layer -- we have 17 classes

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    if not os.path.exists(top_layers_checkpoint_path):
        # first: train only the top layers (which were randomly initialized) i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        #model = multi_gpu_model(model, gpus=3)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        mc_top = ModelCheckpoint(top_layers_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

        model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=top_epochs,
                            validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                            callbacks=[mc_top], class_weight=class_weights)

    model.load_weights(top_layers_checkpoint_path)        
    print ("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")

    if os.path.exists(fine_tuned_checkpoint_path):
        model.load_weights(fine_tuned_checkpoint_path)
        print("Checkpoint '" + fine_tuned_checkpoint_path + "' loaded.")

    # we chose to train the top 2 inception blocks, i.e. we will freeze the first 172 layers and unfreeze the rest:
    for layer in model.layers[:163]:
        layer.trainable = False
    for layer in model.layers[163:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect. we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

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
        callbacks=[mc_fit],
        class_weight=class_weights)


def fine_tune_vgg(frames_folder, model_dir):
    base_model = VGG16(weights='imagenet', include_top=False)
    base_model.summary()
    top_layers_checkpoint_path = os.path.join(model_dir, 'cp.top.best.hdf5')
    fine_tuned_checkpoint_path = os.path.join(model_dir, 'cp.fine_tuned.best.hdf5')

    img_width, img_height = 224, 224
    nb_train_samples = 45450
    nb_validation_samples = 494

    labels = meta.get_train_labels_list()
    class_weights = compute_class_weight('balanced', np.unique(labels), labels)

    top_epochs = 10
    fit_epochs = 65
    batch_size = 64

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

    train_labels = meta.get_images_labels("metadata/training_set.csv")
    test_labels = meta.get_images_labels("metadata/testing_set.csv")

    train_generator = multilabel_flow(frames_folder, train_datagen, train_labels, batch_size=batch_size,
                                      target_size=(img_height, img_width),
                                      sub_folder='training-all')

    validation_generator = multilabel_flow(frames_folder, test_datagen, test_labels, batch_size=batch_size,
                                           target_size=(img_height, img_width),
                                           sub_folder='testing-all')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # predictions = Dense(17, activation='softmax')(x)  # and a logistic layer -- we have 17 classes
    predictions = Dense(17, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    if not os.path.exists(top_layers_checkpoint_path):
        # first: train only the top layers (which were randomly initialized) i.e. freeze all VGG layers
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        mc_top = ModelCheckpoint(top_layers_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)

        model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=top_epochs,
                            validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                            callbacks=[mc_top], class_weight=class_weights)

    model.load_weights(top_layers_checkpoint_path)
    print("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")

    if os.path.exists(fine_tuned_checkpoint_path):
        model.load_weights(fine_tuned_checkpoint_path)
        print("Checkpoint '" + fine_tuned_checkpoint_path + "' loaded.")

    # we chose to train the last convolutional layer
    for layer in model.layers[:25]:
        layer.trainable = False
    for layer in model.layers[25:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect. we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

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
        callbacks=[mc_fit],
        class_weight=class_weights)


def recognize(args):
    frame_dir = args.frame_dir
    model_path = args.model_path
    
    model = load_model(model_path) # Audio tagging

    na_list = []

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        frame_dir,
        target_size=(299, 299),
        color_mode="rgb",
        shuffle = "false",
        class_mode=None,
        batch_size=1)   
   
    labels_indices = [sorted(config.labels, key=str.lower).index(label) for label in config.labels]
    fusion_at = model.predict_generator(test_generator, steps=len(test_generator.filenames))[:, labels_indices]

    for frame_path in test_generator.filenames:
        na, _ = os.path.splitext(os.path.basename(frame_path))
        na = na[1:na.rfind("_")] + ".wav"
        na_list.append(na)

    print("AT shape: %s" % (fusion_at.shape,))
    na_list = [x.encode('utf-8') for x in na_list]
    
    io_task4.at_write_prob_mat_to_csv(
        na_list=na_list, 
        prob_mat=fusion_at, 
        out_path=args.out_dir) # "at_prob_mat.csv.gz"))
            
    print("Prediction finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_finetune_inception = subparsers.add_parser('fine-tune-inception')
    parser_finetune_inception.add_argument('--frames_folder', type=str)
    parser_finetune_inception.add_argument('--model_dir', type=str)

    parser_finetune_vgg = subparsers.add_parser('fine-tune-vgg')
    parser_finetune_vgg.add_argument('--frames_folder', type=str)
    parser_finetune_vgg.add_argument('--model_dir', type=str)

    parser_recognize = subparsers.add_parser('recognize')
    parser_recognize.add_argument('--frame_dir', type=str)
    parser_recognize.add_argument('--model_path', type=str)
    parser_recognize.add_argument('--out_dir', type=str)

    args = parser.parse_args()

    if args.mode == 'fine-tune-inception':
        fine_tune_inception(args.frames_folder, args.model_dir)
    elif args.mode == 'fine-tune-vgg':
        fine_tune_vgg(args.frames_folder, args.model_dir)
    elif args.mode == 'recognize':
        recognize(args)
