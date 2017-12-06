import image_data_reader as idr

image_reader = idr.ImageDataReader('../dataset/sample_images', [])
print(image_reader.next_batch())
