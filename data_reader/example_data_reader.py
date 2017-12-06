import image_data_reader as idr
import numpy as np
import image_mean_calculation as imc


image_reader = idr.ImageDataReader(root_directory='../dataset/sample_images',
                                   mean_path='../dataset/mean.json')
image_reader.next_batch()
#imc.calculate_bgr_channel_mean('../dataset/sample_images','../dataset/mean.json')
