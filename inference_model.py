from keras.optimizers import *
from seq2seqbuilder import Seq2SeqBuilder
from result_visualisation import Inference
import time as time

dataset_type = "train"
model_name = "2018-02-01_16:22:39-2018-02-03_00:22:15"
model_file_name = "./trained_models/" + model_name + ".h5"

# model_file_name = "trained_models/2018-01-18_17:39:24-2018-01-20_18:50:39:image_to_text_gru.h5"
builder = Seq2SeqBuilder()
encoder_model, decoder_model = builder.build_encoder_decoder_inference(model_file_name)

inference = Inference('./dataset/image_embeddings_to_sentence/stories_to_index_'+ dataset_type +'.hdf5',
                      './dataset/vist2017_vocabulary.json', encoder_model, decoder_model)
t = time.time()
inference.predict_all(batch_size=64, references_file_name='',
                      hypotheses_file_name="./results/"+ model_name +"/hypotheses_" +dataset_type + ".txt")
#beam_size = 3
#inference.predict_all_beam_search(batch_size=65, beam_size=beam_size, hypotheses_file_name="./results/"+ model_name +"/hypotheses_" +dataset_type + "_beam"+str(beam_size)+".txt")
print((time.time() - t) / 60.0)
