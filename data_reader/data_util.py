import tensorflow as tf
from glob import glob
import os
from numpy import *
import numpy as np
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.ndimage import filters
from PIL import Image

from tensorflow.contrib.data import Dataset, Iterator

global image_height
global image_width
image_height = image_width = 227
#Input parser that reads one image at a time, transforms it from RGB TO BGR#
#And substracts it with the training dataset mean
def input_parser(img_path,labels):

    img_file=tf.read_file(img_path)
    img_decoded=tf.image.decode_jpeg(img_file,channels=3)
    resized_img=tf.image.resize_image_with_crop_or_pad(img_decoded,image_height,image_width)
    red, green, blue = tf.split(resized_img, num_or_size_splits=3, axis=2)

    red= tf.cast(red, tf.float32)
    blue = tf.cast(blue, tf.float32)
    green = tf.cast(green, tf.float32)

    blue_mean_tf=tf.convert_to_tensor(blue_mean, tf.float32)
    green_mean_tf = tf.convert_to_tensor(green_mean, tf.float32)
    red_mean_tf = tf.convert_to_tensor(red_mean, tf.float32)

    #TODO Why we have axis = 2
    bgr = tf.concat([tf.subtract(blue,blue_mean_tf),
                   tf.subtract(green,green_mean_tf),
                   tf.subtract(red,red_mean_tf)], axis=2)

    return bgr,labels

#Finds pictures in all folders and subfolders and returns their full path names, and their indicies
def get_all_filenames(file_path):
    images_path_names = [y for x in os.walk(file_path) for y in glob(os.path.join(x[0], "*.png"))]
    image_index=[]
    for i in range(len(images_path_names)):
        name=images_path_names[i].split("/")
        nm1=name[len(name)-1].split(".")
        image_index.append(nm1[0])
    return images_path_names,image_index

#Returns a tensor of all the image tensors, and their lables(indecies)
def get_data_from_path(file_path,bgr_path,batch_size=1):
    img_pathnames, img_indicies = get_all_filenames(file_path)
    tr_data = tf.data.Dataset.from_tensor_slices((img_pathnames, img_indicies))
    global blue_mean,green_mean,red_mean
    blue_mean,green_mean,red_mean=get_bgr_channel_mean(bgr_path)
    blue_mean=np.reshape(blue_mean,[image_height,image_width,1])
    green_mean = np.reshape(green_mean, [image_height, image_width, 1])
    red_mean = np.reshape(red_mean, [image_height, image_width, 1])
    #tr_data = tr_data.map(input_parser)
    tr_data = tr_data.map(input_parser, num_threads=8, output_buffer_size=100 * batch_size)
    tr_data = tr_data.batch(2)

    return tr_data


def calculate_bgr_channel_mean(files_path, save_path):
        images_path, _ = get_all_filenames(files_path)
        red_channel = np.zeros([image_height, image_width], float32)
        green_channel = np.zeros([image_height, image_width], float32)
        blue_channel = np.zeros([image_height, image_width], float32)
        num_images = len(images_path)

        for img_path in images_path:
            # t = time.time()
            image = (imread(img_path)[:, :, :3]).astype(float32)

            #TODO: How is the resize implemented, does this image do center crop?
            image = resize(image, [image_height, image_width, 3])

            red_channel = np.add(red_channel, image[:, :, 0])
            green_channel = np.add(green_channel, image[:, :, 1])
            blue_channel = np.add(blue_channel, image[:, :, 2])
            # print(time.time()-t)

        red_channel = np.divide(red_channel, num_images)
        blue_channel = np.divide(blue_channel, num_images)
        green_channel = np.divide(green_channel, num_images)

        np.savez(save_path, blue_channel = blue_channel, green_channel=green_channel, red_channel=red_channel)

def get_bgr_channel_mean(bgr_path):
    file=np.load(bgr_path)
    red_mean = file['red_channel']
    blue_mean= file['blue_channel']
    green_mean=file['green_channel']
    return blue_mean, green_mean, red_mean




# tr_data,_=get_data("/Users/wf-markosmilevski/PycharmProjects/alexnet","/Users/wf-markosmilevski/PycharmProjects/alexnet/test.npz")
# iterator=Iterator.from_structure(tr_data.output_types,tr_data.output_shapes)
# next_element=iterator.get_next()
#
# trainig_init_op=iterator.make_initializer(tr_data)
# with tf.Session() as sess:
#      sess.run(trainig_init_op)
#      while True:
#          try:
#              elem=sess.run(next_element)
#          except tf.errors.OutOfRangeError:
#              print("End of trainig dataset.")
#              break

