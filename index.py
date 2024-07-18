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

# this function creates a tf.data.Dataset from the directory structure.
#the folder train has subfolders, and  each subfolder represents a different class.
#tf.keras.preprocessing creates a ds from the retrieved folder structure
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

#initialization of dictionarty where we store one image per type,and iterate over the types
imgs_per_type = {type: None for type in alz_types_train}

#for each batch in the train ds,using label the type is determined of the image
#if an img has not been stored yet it is stored in the images per type
for images, labels in train_dataset: #iteration over batches,each iteration retrieves a batch of images and labels
    for image, label in zip(images, labels): #zip pairs an image with a label
        type = alz_types_train[label] #determines the type,alz_types_train is a list of class names. label is an integer index, so alz_types_train[label] gives the type for the current image.
        if imgs_per_type[type] is None: #checks if the type is already stored
            imgs_per_type[type] = (image.numpy(), type)
        if all(imgs_per_type.values()):
            break
            #here we check if we have an image from every type,and if we have we break the loop
    if all(imgs_per_type.values()):
        break

plt.figure(figsize=(10, 10))
for i, (image, type) in enumerate(imgs_per_type.values()):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.astype("uint8"))
    plt.title(type)
    plt.axis("off")
plt.show()
