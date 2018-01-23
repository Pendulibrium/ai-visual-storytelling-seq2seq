from keras.optimizers import *
from seq2seqbuilder import Seq2SeqBuilder
from result_visualisation import Inference
import time as time

builder = Seq2SeqBuilder()
encoder_model, decoder_model = builder.build_encoder_decoder_inference(
    "trained_models/2018-01-18_17:39:24-2018-01-20_18:50:39:image_to_text_gru.h5")

inference = Inference('./dataset/image_embeddings_to_sentence/stories_to_index_valid.hdf5',
                      './dataset/vist2017_vocabulary.json', encoder_model, decoder_model)
t = time.time()
inference.predict_all(batch_size=5)
print((time.time() - t) / 60.0)
