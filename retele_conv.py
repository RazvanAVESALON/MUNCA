import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import pathlib as pt

import yaml
# pip install pyyaml

config = None
with open('config.yml') as f: # reads .yml/.yaml files
    config = yaml.load(f)

print(type(config))
print(config)
print(f"n1 = {config['net']['n1']}", type(config['net']['n1']))
print(f"conv kernel = {config['net']['conv']}", type(config['net']['conv']))
print(f"opt = {config['train']['opt']}", type(config['train']['opt']))
print(f"lr = {config['train']['lr']}", type(config['train']['lr']))


# Date de training
DATA_PATH = pt.Path(r"D:\\ai intro\AI intro\7. Retele Complet Convolutionale\DA")
VAL_SPLIT = 0.2
BATCH_SIZE = config['train']['bs']

# Atentie!!!
# image_dataset_from_directory(main_directory, labels='categorical') returneaza imagini din subdirectoarele `covid` si `normal`
# impreuna cu etichetele 0 and 1 (0 corespunde cu `covid` si 1 corespunde cu `normal`, adica ordine alfabetica)
# Pentru a schimba ordinea se va da o lista explicita de `class_names`

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  DATA_PATH,
  validation_split=0.2,
  subset="training",
  shuffle=True,
  seed=123,
  image_size=config['net']['img'],
  class_names=[ 'Normal', 'COVID'],
  label_mode='categorical', # Codificam clasele cu un vector de probabilitati. 
  batch_size=BATCH_SIZE)

# Date de testing
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  DATA_PATH,
  validation_split=VAL_SPLIT,
  subset="validation",
  seed=123,
  shuffle=False,
  image_size=config['net']['img'],
  class_names=['Normal', 'COVID'],
  label_mode='categorical',
  batch_size=BATCH_SIZE)


x_train, y_train = next(iter(train_ds))
# Parcurgand obiectul de tip tf.Dataset obtinem tensori de imagini si etichete cu shape-ul [batch_size, height, width, nr_channels]
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)

x_test, y_test = next(iter(test_ds))
# La fel si pentru setul de test [batch_size, height, width, nr_channels]
print('x_test.shape: ', x_test.shape)
print('y_test.shape: ', y_test.shape)

# Numele etichetelor poate fi determinat cu ajutorul atributului `class_names`
class_names = train_ds.class_names
print(class_names)
# Vizualizarea imaginilor
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[np.argmax(labels[i])])
    plt.axis("off")
# plt.show()


# Normalizarea setului de date
# Se aplica un start the normalizare dataset-ul de antrenare si testare
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_train_ds))
first_image = image_batch[0]

# Imaginile vor avea valori in intervalul `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Atentie! Normalizarea se face si pentru datasetul de testare
normalized_test_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Dimensiunea imaginilor
image_dim = x_train[0].shape

# Determinam numarul de clase unice. Acestea sunt one-hot encoded
nr_clase = len(np.unique(y_train))
print(f'Numarul de clase unice: {nr_clase}')

# API secvential
n1 = config['net']['n1']
n2 = config['net']['n2']
n3 = config['net']['n2']

# Definim arhitectura retelei convolutionale
model = Sequential()
model.add(Conv2D(n1, config['net']['conv'], activation='relu', padding='valid', input_shape=image_dim))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(n2, config['net']['conv2'], activation='relu', padding='valid'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(n3, activation='relu'))
model.add(Dense(nr_clase, activation='softmax'))
# model.summary()

# Antrenarea modelului
loss = tf.keras.losses.CategoricalCrossentropy()  # Functia loss
opt = tf.keras.optimizers.Adam(learning_rate=config['train']['lr']) # Optimizatorul
model.compile(optimizer=opt, loss=loss)  # Compilarea modelului

# Calcularea nr total de imagini. 
# daca ver tensorflow < 2.4.0
# Se cauta toate tipurile de imagini din locatia DATA_PATH 
# file_types = ['.jpg', '.jpeg', '.png']
# dataset_size = len([p for p in DATA_PATH.rglob("*") if p.suffix in file_types]) # nr total imagini

# Calcularea nr de imagini in setul de antrenare si cel de testare
# test_size = int(dataset_size * VAL_SPLIT)
# train_size = dataset_size - test_size

# tensorflow v > 2.4.0
train_size = len(train_ds.file_paths) 
test_size = len(test_ds.file_paths)
dataset_size = train_size + test_size

print(f"\nNr total imagini dataset: {dataset_size}")
print(f"Train: nr imagini: {train_size} Test: nr imagini: {test_size}")

nr_epochs = config['train']['n_epochs']
nr_train_iterations = train_size // BATCH_SIZE
print(f"Nr iteratii/epoca: {nr_train_iterations}\n")


# Pentru optimizarea citirii de pe disk https://www.tensorflow.org/tutorials/load_data/images#configure_the_dataset_for_performance
AUTOTUNE = tf.data.AUTOTUNE
normalized_train_ds = normalized_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Bucla de antrenare. Se va folosi datasetul normalizat!!
evolutie_acc = []
evolutie_loss = []

for ep in range(nr_epochs):
    predictions = []
    true_labels = []
    loss_per_epoch = 0

    for step, (batch_data, batch_labels) in enumerate(normalized_train_ds):
        # batch_data = x_train[it * batch_size: it * batch_size + batch_size, :]
        # batch_labels = y_train_labels[it * batch_size: it * batch_size + batch_size]
        # print(batch_data.shape, batch_labels.shape)

        err = model.train_on_batch(batch_data, batch_labels)
        loss_per_epoch = loss_per_epoch + err
        pred_probs = model.predict_on_batch(batch_data)

        pred_label = np.argmax(pred_probs, axis=1)

        predictions = np.concatenate((predictions, pred_label))
        true_labels = np.concatenate((true_labels, np.argmax(batch_labels, axis=1)))

    evolutie_loss.append(loss_per_epoch/nr_train_iterations)

    # Calculam acuratetea
    acc = np.sum(predictions == true_labels) / len(predictions)
    evolutie_acc.append(acc)
    print(f'Epoch {ep}: Loss: {loss_per_epoch / nr_train_iterations} Acc: {acc * 100}')

    # Salvam ponderile modelului dupa fiecare epoca
    model.save_weights('my_model')
    train_iter = iter(train_ds)

# Testarea modelului
# Incarcam ponderile modelul antrenat
model.load_weights('my_model')

predictions = []
test_labels = []
for step, (batch_data, batch_labels) in enumerate(normalized_test_ds):
    # batch_data = x_test[it * batch_size: it * batch_size + batch_size, :]
    current_pred_probs = model.predict_on_batch(batch_data)
    current_pred_probs = np.argmax(current_pred_probs, axis=1)

    predictions = np.concatenate((predictions, current_pred_probs))
    test_labels = np.concatenate((test_labels, np.argmax(batch_labels, axis=1)))


acc = np.sum(predictions == test_labels) / len(predictions)
print(f'\nAcuratetea la test este {acc * 100}')
