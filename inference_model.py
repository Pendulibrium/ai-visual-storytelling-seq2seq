from keras.optimizers import *
from seq2seqbuilder import Seq2SeqBuilder
from result_visualisation import Inference
import time as time

dataset_type = "valid"
model_name = "2018-01-29_00:37:26-2018-01-31_00:01:42"
model_file_name = "./trained_models/" + model_name + ".h5"

builder = Seq2SeqBuilder()
encoder_model, decoder_model = builder.build_encoder_decoder_inference(model_file_name)

inference = Inference('./dataset/image_embeddings_to_sentence/stories_to_index_'+ dataset_type +'.hdf5',
                      './dataset/vist2017_vocabulary.json', encoder_model, decoder_model)
t = time.time()
inference.predict_all(batch_size=64, references_file_name='',
                      hypotheses_file_name="./results/"+ model_name +"/hypotheses_" +dataset_type + ".txt")
print((time.time() - t) / 60.0)
