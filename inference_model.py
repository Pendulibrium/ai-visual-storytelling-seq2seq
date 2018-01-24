from keras.optimizers import *
from seq2seqbuilder import Seq2SeqBuilder
from result_visualisation import Inference
import time as time

model_name = "2018-01-20_22:10:16-2018-01-21_09:53:21:image_to_text_gru.h5"
builder = Seq2SeqBuilder()
encoder_model, decoder_model = builder.build_encoder_decoder_inference("trained_models/" + model_name)

inference = Inference('./dataset/image_embeddings_to_sentence/stories_to_index_valid.hdf5',
                      './dataset/vist2017_vocabulary.json', encoder_model, decoder_model)
t = time.time()
inference.predict_all(batch_size=64, references_file_name='',
                      hypotheses_file_name="2018-01-20_22:10:16-2018-01-21_09:53:21_valid")
print((time.time() - t) / 60.0)
