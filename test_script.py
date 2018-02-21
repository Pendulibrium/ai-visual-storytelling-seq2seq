from keras.optimizers import *
from seq2seqbuilder import Seq2SeqBuilder
from result_visualisation import Inference
import time as time
from keras import backend as K
from model_data_generator import ModelDataGenerator
import json
import h5py
import matplotlib.pyplot as plt
import numpy as np

#The learning phase flag is a bool tensor (0 = test, 1 = train) to be passed as input to any Keras function that uses a different behavior at train time and test time.
K.set_learning_phase(0)
dataset_type = "valid"
model_name = "2018-02-09_15:30:08-2018-02-10_01:04:10"
model_file_name = "./trained_models/" + model_name + ".h5"

# model_file_name = "trained_models/2018-01-18_17:39:24-2018-01-20_18:50:39:image_to_text_gru.h5"
builder = Seq2SeqBuilder()
im_model, sent_model = builder.build_encoder_decoder_inference_from_file(model_file_name,
                                                                                 include_sentence_encoder=True)
vocab_json = json.load(open('./dataset/vist2017_vocabulary.json'))
dataset_file = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_' + "valid" + '.hdf5', 'r')
data_generator = ModelDataGenerator(dataset_file, vocab_json, 5)

count=0

for batch in data_generator.multiple_samples_per_story_generator(reverse=False, only_one_epoch=True, last_k=3):
    encoder_batch_input_data = batch[0][0]
    encoder_batch_sentence_input_data = batch[0][1]
    original_sentences_input = batch[0][2]
    count+=1
    image_encoder_data = encoder_batch_input_data[0]
    image_encoder_data = image_encoder_data.reshape((1, image_encoder_data.shape[0], image_encoder_data.shape[1]))
    image_embeddings = im_model.predict([image_encoder_data, encoder_batch_sentence_input_data[0]])
    sentence_embeddings = sent_model.predict([image_encoder_data, encoder_batch_sentence_input_data[0]])
    print(image_embeddings.shape)
    print(sentence_embeddings.shape)
    print(image_embeddings[0,0:20])
    print(sentence_embeddings[0, 0:20])
    #print(np.histogram(image_embeddings[0, :]))
    #print(np.histogram(sentence_embeddings[0, :]))
    if count%5==0:
        break
    #plot_url = py.plot_mpl(fig, filename='im-basic-histogram')
    #
    # plt.hist(sentence_embeddings)
    # plt.title("Sentence embeddings Histogram")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    #
    # fig = plt.gcf()
    #
    # plot_url = py.plot_mpl(fig, filename='sent-basic-histogram')




#print(np.dot(image_embeddings[0,:], sentence_embeddings[0,:]))