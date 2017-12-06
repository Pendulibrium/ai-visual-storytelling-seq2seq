import cv2
import numpy as np

# Returns resized image in aspect to ratio
def resize_in_aspect_to_ration(image,image_size):
        height = image.shape[0]
        width = image.shape[1]
        ratio = float(height) / float(width)
        if ratio > 1:
            new_width = image_size[0]
            new_height = new_width * ratio
        else:
            new_height = image_size[1]
            new_width = new_height / ratio

        dim = (int(new_width), int(new_height))
        image1 = cv2.resize(image, dim)
        return image1

    # Returns center cropped image
def center_crop_image(image):
        height = image.shape[0]
        width = image.shape[1]

        if height >= width:
            center = height / 2
            left = center - width / 2
            right = center + width / 2
            crop_img = image[left:right + 1, 0:width]
        else:
            center = width / 2
            left = center - height / 2
            right = center + height / 2
            crop_img = image[0:height, left:right + 1]

        return crop_img