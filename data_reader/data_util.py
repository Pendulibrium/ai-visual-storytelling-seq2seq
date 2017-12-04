import tensorflow as tf
from glob import glob
import os
import read_means_file as rmf
from tensorflow.contrib.data import Dataset, Iterator


#Input parser that reads one image at a time, transforms it from RGB TO BGR#
#And substracts it with the training dataset mean
def input_parser(img_path,labels):
    img_file=tf.read_file(img_path)
    img_decoded = tf.image.decode_png(img_file, channels=3)
    #img_decoded=tf.image.decode_jpeg(img_file,channels=3)
    resized_img=tf.image.resize_image_with_crop_or_pad(img_decoded,277,277)

    red,green,blue=tf.split(resized_img,num_or_size_splits=3,axis=2)
    #assert red.get_shape().as_list()[1:]==[227,227,1]
    #assert green.get_shape().as_list()[1:] == [277, 277, 1]
    #assert blue.get_shape().as_list()[1:] == [277, 277, 1]
    #tuka treba da se presmetaat blue-mean(blue)
    red= tf.cast(red, tf.float32)
    blue = tf.cast(blue, tf.float32)
    green = tf.cast(green, tf.float32)

    bgr=tf.concat([tf.subtract(blue,bgr_means[0]),
                   tf.subtract(green,bgr_means[1]),
                   tf.subtract(red,bgr_means[2])],axis=2)
    #bgr = tf.concat([blue,green,red],axis=2)
    #assert bgr.get_shape().as_list()[1:] == [277, 277, 3]

    return bgr,labels

#Finds pictures in all folders and subfolders and returns their full path names, and their indicies
def get_data_from_path(mypath):
    images_path_names = [y for x in os.walk(mypath) for y in glob(os.path.join(x[0], "*.png"))]
    image_index=[]
    for i in range(len(images_path_names)):
        name=images_path_names[i].split("/")
        nm1=name[len(name)-1].split(".")
        image_index.append(int(nm1[0]))
    return images_path_names,image_index

#Returns a tensor of all the image tensors, and their lables(indecies)
def get_data(mypath,batch_size=1):
    train_imgs, train_labels = get_data_from_path(mypath)
    tr_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
    global bgr_means
    #vo get_bgr_means treba imeto na checkpointot da se smeni
    bgr_means=rmf.get_bgr_channel_mean()
    tr_data = tr_data.map(input_parser)
    #tr_data = tr_data.map(input_parser,num_threads=8,output_buffer_size=100*batch_size)

    return tr_data,train_labels


#
# tr_data,_=get_data("/Users/wf-markosmilevski/PycharmProjects/alexnet")
#
#
# iterator=Iterator.from_structure(tr_data.output_types,tr_data.output_shapes)
#
# next_element=iterator.get_next()
#
# trainig_init_op=iterator.make_initializer(tr_data)
# with tf.Session() as sess:
#      sess.run(trainig_init_op)
#      while True:
#          try:
#              elem=sess.run(next_element)
#              print(elem)
#          except tf.errors.OutOfRangeError:
#              print("End of trainig dataset.")
#              break