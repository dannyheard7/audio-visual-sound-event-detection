import argparse
import os

from keras import Model
from keras.applications import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.training_utils import multi_gpu_model


def fine_tune_inception(frames_folder, model_dir):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    top_layers_checkpoint_path = os.path.join(model_dir, 'cp.top.best.hdf5')
    fine_tuned_checkpoint_path = os.path.join(model_dir, 'cp.fine_tuned.best.hdf5')
    new_extended_inception_weights = os.path.join(model_dir, 'final_weights.hdf5')
    
    print(top_layers_checkpoint_path)
    
    train_data_dir = os.path.join(frames_folder, 'training')
    validation_data_dir = os.path.join(frames_folder, 'testing')
    img_width, img_height = 299, 299
    nb_train_samples = 41497
    nb_validation_samples = 396

    top_epochs = 10
    fit_epochs = 50
    batch_size = 24

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

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
    x = Dense(1024, activation='relu')(x) # let's add a fully-connected layer
    predictions = Dense(17, activation='softmax')(x) # and a logistic layer -- we have 17 classes

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    if not os.path.exists(top_layers_checkpoint_path):
        # first: train only the top layers (which were randomly initialized) i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        #model = multi_gpu_model(model, gpus=3)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        mc_top = ModelCheckpoint(top_layers_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

        model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=top_epochs,
                            validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size, callbacks=[mc_top])

    else:
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

    #multi_model = multi_gpu_model(model, gpus=3)
    # we need to recompile the model for these modifications to take effect. we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

    # Save the model after every epoch.
    mc_fit = ModelCheckpoint(fine_tuned_checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--frames_folder', type=str)
    parser.add_argument('--model_dir', type=str)

    args = parser.parse_args()

    fine_tune_inception(args.frames_folder, args.model_dir)
