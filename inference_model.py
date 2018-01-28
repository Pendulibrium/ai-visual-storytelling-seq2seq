from keras.optimizers import *
from seq2seqbuilder import Seq2SeqBuilder
from result_visualisation import Inference
import time as time

model_name = "2018-01-25_10:05:24-2018-01-27_22:34:08:image_to_text_gru.h5"
builder = Seq2SeqBuilder()
encoder_model, decoder_model = builder.build_encoder_decoder_inference("trained_models/" + model_name)

inference = Inference('./dataset/image_embeddings_to_sentence/stories_to_index_train.hdf5',
                      './dataset/vist2017_vocabulary.json', encoder_model, decoder_model)
t = time.time()
inference.predict_all(batch_size=64, references_file_name='',
                      hypotheses_file_name="./results/2018-01-25_10:05:24-2018-01-27_22:34:08/hypotheses_train.txt")
print((time.time() - t) / 60.0)
