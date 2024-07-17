import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

train = r'D:\Diplomska\DiplomskiTrud\train'
test = r'D:\Diplomska\DiplomskiTrud\test'

img_height = 150
img_width = 150
batch_size = 32

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train,  # training data set read earlier
    seed=123,  # this is for randomly shuffling the data
    image_size=(img_height, img_width),  # fixed image height and width for all images in the data
    batch_size=batch_size  # number of images that are processed in one batch
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test,  # testing data set read earlier
    seed=123,  # this is for randomly shuffling the data
    image_size=(img_height, img_width),  # fixed image height and width for all images in the data
    batch_size=batch_size  # number of images that are processed in one batch
)

alz_types_train = train_dataset.class_names
alz_types_test = test_dataset.class_names

print('Training set:', alz_types_train)
print('Testing set:', alz_types_test)

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(alz_types_train[labels[i]])
        plt.axis("off")
plt.show()
