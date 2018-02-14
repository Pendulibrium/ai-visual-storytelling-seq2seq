from keras.optimizers import *
from seq2seqbuilder import Seq2SeqBuilder
from result_visualisation import Inference
import time as time
from keras import backend as K


#The learning phase flag is a bool tensor (0 = test, 1 = train) to be passed as input to any Keras function that uses a different behavior at train time and test time.
K.set_learning_phase(0)
dataset_type = "valid"
model_name = "2018-02-13_14:50:10-2018-02-14_10:10:33"
model_file_name = "./trained_models/" + model_name + ".h5"

# model_file_name = "trained_models/2018-01-18_17:39:24-2018-01-20_18:50:39:image_to_text_gru.h5"
builder = Seq2SeqBuilder()
encoder_model, decoder_model = builder.build_encoder_decoder_inference_from_file(model_file_name,
                                                                                 include_sentence_encoder=True)

inference = Inference('./dataset/image_embeddings_to_sentence/stories_to_index_' + dataset_type + '.hdf5',
                      './dataset/vist2017_vocabulary.json', encoder_model, decoder_model)
t = time.time()
# inference.predict_all(batch_size=64, references_file_name='',
#                       hypotheses_file_name="./results/"+ model_name +"/hypotheses_" +dataset_type + ".txt")
# beam_size = 10
# inference.predict_all_beam_search(batch_size=600, beam_size=beam_size, hypotheses_file_name="./results/"+ model_name +"/hypotheses_" +dataset_type + "_beam"+str(beam_size)+".txt")
inference.predict_all(batch_size=50, references_file_name='',
                      hypotheses_file_name="./results/" + model_name + "/hypotheses_" + dataset_type + ".txt",
                      sentence_embedding=True)
print((time.time() - t) / 60.0)

# a = [2101 - we had a great time , 2110 - he was so excited to see his brother [male] .]
