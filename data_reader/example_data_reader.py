import image_data_reader as idr
import numpy as np
import image_mean_calculation as imc
import h5py
import time

#image_reader = idr.ImageDataReader(root_directory='../dataset/sample_images',
#                                   mean_path='../dataset/mean.json')
#image_reader.next_batch()
#imc.calculate_bgr_channel_mean('../dataset/vist_dataset/training_data/train-img/images','../dataset/mean.json')


t=time.time()
keys=range(1,170000)
mat=np.random.randn(170000,4096)
print((time.time()-t)/60.0)

dict={}
t=time.time()

data_file=h5py.File('file.hdf5','w')
for i in keys:
    data_file.create_dataset(str(i), data=mat[i,:].tolist())


data_file.close
print((time.time()-t)/60.0)

f=h5py.File('file.hdf5','r')
print("keys: %s" % f.keys())
