import tensorflow as tf

#Returns BGR channel mean, that was saved with the calculate_bgr_channel function
def get_bgr_channel_mean(checkpoint_path="/Users/wf-markosmilevski/PycharmProjects/alexnet/test_model.ckpt"):
    tf.reset_default_graph()

    green_mean = tf.get_variable("green_mean", shape=[277, 277, 1])
    blue_mean = tf.get_variable("blue_mean", shape=[277, 277, 1])
    red_mean = tf.get_variable("red_mean", shape=[277, 277, 1])

    with tf.Session() as sess:
        saver=tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        #print(sess.run(green_mean))
        #print("Model restored.")

    return (blue_mean,green_mean,red_mean)
