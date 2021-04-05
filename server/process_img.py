import cv2
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras import backend as K



def process_img(img):
	K.clear_session()
    # returns a grayscale version of the given image
    # load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	features = dict()
	# filename = directory
	image = load_img('F:\\single photo traffic server\\server_example\\'+img, target_size=(224, 224))

	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the models
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)

	return feature