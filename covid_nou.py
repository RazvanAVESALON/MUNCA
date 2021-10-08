import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import tensorflow_datasets as tfds
from numpy import unique
from numpy import argmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import yaml
# pip install pyyaml

config = None
with open('config.yml') as f: # reads .yml/.yaml files
    config = yaml.load(f)

print(type(config))
print(config)
print(f"n1 = {config['net']['n1']}", type(config['net']['n1']))
print(f"opt = {config['train']['opt']}", type(config['train']['opt']))
print(f"lr = {config['train']['lr']}", type(config['train']['lr']))


def plot_acc_loss(result):
    acc = result.history['accuracy']
    loss = result.history['loss']
    val_acc = result.history['val_accuracy']
    val_loss = result.history['val_loss']
    pyplot.figure(figsize=(15, 5))
    pyplot.subplot(121)
    pyplot.plot(acc, label='Train')
    pyplot.plot(val_acc, label='Validation')
    pyplot.title('Accuracy', size=15)
    pyplot.legend()
    pyplot.grid(True)
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    
    pyplot.subplot(122)
    pyplot.plot(loss, label='Train')
    pyplot.plot(val_loss, label='Validation')
    pyplot.title('Loss', size=15)
    pyplot.legend()
    pyplot.grid(True)
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    
    pyplot.show()
    

dataset_dir  = r"D:\ai intro\dataset"
IMAGE_SIZE = config['net']['img']
BATCH_SIZE = config['train']['bs']



train_datagen = ImageDataGenerator( rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)     

train_batches = train_datagen.flow_from_directory(dataset_dir + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode="binary")

validation_batches = validation_datagen.flow_from_directory(dataset_dir + '/val',
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode="binary")


from matplotlib import pyplot
x_test, y_test = next(train_batches)
print(x_test.shape)
print(y_test)

labels = {0: 'COVID', 1: 'Normal'}

fig = pyplot.figure(figsize=(32,32))
for i in range(x_test.shape[0]):
    ax = fig.add_subplot(int(x_test.shape[0]/2), int(x_test.shape[0]/2), i+1)
    ax.title.set_text(labels[y_test[i]])
    pyplot.imshow(x_test[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

in_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=in_shape)
conv_base.summary()
n1=config['net']['n1']


model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(n1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
conv_base.trainable = False

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=config['train']['lr']), metrics=['accuracy'])

NUM_EPOCHS = config['train']['n_epochs']
history = model.fit(train_batches, steps_per_epoch = config['train']['steps_per_epoch'], validation_data = validation_batches, validation_steps = config['net']['steps'], epochs= NUM_EPOCHS)

model.save('damn.h5')   

plot_acc_loss(history)
test_generator = validation_datagen.flow_from_directory(dataset_dir + '/test', target_size=config['net']['img'], batch_size=config['train']['bs'], class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=config['net']['steps'])
print('test acc:', test_acc)