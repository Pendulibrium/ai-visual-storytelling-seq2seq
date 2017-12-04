import tensorflow as tf
import data_util as du
import time

#Calculates a matrix which represent the mean for each channel
#Return 3 matrices
def calculate_bgr_channel_mean(sess,mypath="/Users/wf-markosmilevski/PycharmProjects/alexnet",
                         save_path="/Users/wf-markosmilevski/PycharmProjects/alexnet/test_model.ckpt"):

    #mypath="/Users/wf-markosmilevski/PycharmProjects/alexnet"
    images_path,_=du.get_data_from_path(mypath)
    num_images=tf.constant(len(images_path),tf.float32)
    red_sum=tf.zeros([277,277,1],tf.uint8)
    green_sum = tf.zeros([277, 277, 1], tf.uint8)
    blue_sum = tf.zeros([277, 277, 1], tf.uint8)
    for img_path in images_path:
        t = time.time()
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_png(img_file, channels=3)
        # img_decoded=tf.image.decode_jpeg(img_file,channels=3)
        resized_img = tf.image.resize_image_with_crop_or_pad(img_decoded, 277, 277)

        red, green, blue = tf.split(resized_img, num_or_size_splits=3, axis=2)
        red_sum=tf.add(red,red_sum)
        blue_sum = tf.add(blue, blue_sum)
        green_sum = tf.add(green, green_sum)
        print(time.time()-t)


    red_sum=tf.cast(red_sum,tf.float32)
    blue_sum = tf.cast(blue_sum, tf.float32)
    green_sum = tf.cast(green_sum, tf.float32)

    blue_mean=tf.Variable(tf.divide(blue_sum,num_images),name="blue_mean")
    green_mean=tf.Variable(tf.divide(green_sum,num_images),name="green_mean")
    red_mean = tf.Variable(tf.divide(red_sum, num_images),name="red_mean")
    saver=tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    save_path=saver.save(sess,save_path)
    print("Model saved in file: %s" % save_path)

#k1=get_bgr_channel_mean()
# blue_mean=tf.Variable(k1,name="blue_mean")
# green_mean=tf.Variable(k2,name="green_mean")
# red_mean=tf.Variable(k3,name="red_mean")
#
# init=tf.global_variables_initializer()
#
# saver=tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path=saver.save(sess,"/Users/wf-markosmilevski/PycharmProjects/alexnet/test_model.ckpt")
#     print("Model saved in file: %s" % save_path)


