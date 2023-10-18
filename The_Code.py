import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import uuid #Generate unique name

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf


# Set CUDA_VISIBLE_DEVICES to specify GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Replace with the GPU device IDs you want to use



#Set GPU growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

#Folder structure
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)


# #Taking care of the Negatives
# for directory in os.listdir('lfw'):
# 	for each_person in os.listdir(os.path.join('lfw', directory)):
# 		TEMP_PATH = os.path.join('lfw', directory, each_person)
# 		NEG_PATH_ = os.path.join(NEG_PATH, each_person)
# 		os.replace(TEMP_PATH, NEG_PATH_)


# #Collecting Positive ans anchor images
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
# 	ret, frame = cap.read()
# 	frame = frame[200:200+450, 200:200+300, :]

# 	#Collecting Anchor images
# 	if cv2.waitKey(1) & 0XFF == ord('a'):
# 		imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
# 		cv2.imwrite(imgname, frame)

# 	#Collecting Positive images
# 	if cv2.waitKey(1) & 0XFF == ord('p'):
# 		imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
# 		cv2.imwrite(imgname, frame)


# 	#Show into the screen
# 	cv2.imshow("Image", frame)

# 	if cv2.waitKey(1) & 0XFF == ord('q'):
# 		break

# #We release the webcam
# cap.release()
# cv2.destroyAllWindows()



## Preprocessing

#Creating tensorflow Datasets
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)


#Scale and Resize
def preprocess(file_path):
	byte_img = tf.io.read_file(file_path)
	#decode the jpg images
	image = tf.io.decode_jpeg(byte_img)
	image = tf.image.resize(image, (100,100))
	image = image / 255.0

	return image


# Labelled dataset
negatives = tf.data.Dataset.zip( (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones( len(anchor) )) ) )
positives = tf.data.Dataset.zip( (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros( len(anchor) ))) )
data = positives.concatenate(negatives)


def preprocess_twin(input_img, output_img, label):
	return (preprocess(input_img), preprocess(output_img), label)

# Data loader
data = data.map(preprocess_twin) #Preprocess the whole dataset
data = data.cache()
data = data.shuffle(buffer_size=10000)

#Train partition
train_data = data.take(round(len(data)*0.7)) #Take 70%
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

#Test partition
test_data = data.skip(round(len(data)*0.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


##Model Conception

def make_embedding():
	#Check page 4 of the Siamese NN paper
	inp =  Input(shape = (100, 100, 3), name = 'input_img')

	c1 = Conv2D(64, (10,10), activation='relu')(inp)
	m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

	c2 = Conv2D(128, (7,7), activation='relu')(m1)
	m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

	c3 = Conv2D(128, (4,4), activation='relu')(m2)
	m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

	c4 = Conv2D(256, (4,4), activation='relu')(m3)
	f1 = Flatten()(c4)
	d1 = Dense(4096, activation='sigmoid')(f1)

	#Creating the model
	return Model(inputs=[inp], outputs=[d1], name='embedding')

embd = make_embedding()
#mod.summary()


# L1 Layer - inherit from Layer
class L1Dist(Layer):
	def __init__(self, **kwargs):
		super().__init__()

	def call(self, input_embedding, validation_embedding):
		return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model():
	#check figure 3 of the paper
	anchor_img = Input(name='anchor_img', shape=(100,100,3))
	validation_img = Input(name='validation_img', shape=(100, 100, 3))

	siamese_layer = L1Dist()
	siamese_layer._name = 'distance'
	distances = siamese_layer(embd(anchor_img), embd(validation_img))

	#The last layer
	classifier = Dense(1, activation='sigmoid')(distances)

	return Model(inputs=[anchor_img, validation_img], outputs=classifier, name='Siamese_NN')

siamese_model = make_siamese_model()
siamese_model.summary()




## Training
#Defining the loss function and the optimizer
bin_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) #Learning Rate = 1e-4

## Training checkpoints - so that we can resume
checkPoint_dir = './training_checkPoints'
checkPoint_prefix = os.path.join(checkPoint_dir, 'ckpt')
tf.train.Checkpoint(opt=opt, siames_model=siamese_model)


@tf.function  #conversion into a TensorFlow computation graph
def train_step(batch):
	#compute gradients of the loss with respect to the model's parameters
	with tf.GradientTape() as tape:
		x = batch[:2] 
		y = batch[2]

		#Forward Prop
		yhat = siamese_model(x, training=True)
		# Compute Loss function
		loss = bin_cross_loss(y, yhat)

	print(loss)
	#Back prop
	grad = tape.gradient(loss, siamese_model.trainable_variables)
	#siamese_model.trainable_variables represents a list of all the trainable parameters
	#Updates the parameters
	opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

	return loss


#General Training loop
def train(data, EPOCH):
	for epoch in range(1, EPOCH+1): #we dont like 0 indexiation (in this case)
		print(f"\n Epoch: {epoch}/{EPOCH}")
		progbar = tf.keras.utils.Progbar(len(data))

		#Considering each batch
		for idx, batch in enumerate(data):
			train_step(batch)
			progbar.update(idx+1)

		#And saving the checkpoints !
		if epoch % 10 == 0:
			checkpoint.save(file_prefix=checkPoint_prefix)


train(train_data, 50)


## It is time to evaluate the model
# We will consider recall and precision
r = Recall()
p = Precision()

for test_anchor, test_validation, y_true in test_data.as_numpy_iterator():
	yhat = siamese_model.predict([test_anchor, test_validation])
	r.update_state(y_true, yhat)
	p.update_state(y_true, yhat)

print(r.result().numpy(), p.result().numpy()) 



#Save the model
#siamese_model.save('siamese_model.h5')