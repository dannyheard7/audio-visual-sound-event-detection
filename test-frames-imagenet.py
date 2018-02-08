import glob, os

import config
from meta import load_sound_event_classes
from FileIO import get_class_directory

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np


def test_frame(path):
	img = image.load_img(path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	preds = model.predict(x)
	# decode the results into a list of tuples (class, description, probability)
	# (one such list for each sample in the batch)
	print('Predicted:', decode_predictions(preds, top=3)[0])


model = InceptionV3(weights='imagenet')
class_labels = load_sound_event_classes()


for class_label, class_name in class_labels.items():
	print(class_name)
	frame_folder = get_class_directory(config.video_training_frames_location, class_name)

	os.chdir(frame_folder)
	for file in glob.glob("*" + config.video_frames_extension):
		test_frame(file)


