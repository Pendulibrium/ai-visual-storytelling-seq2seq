from glob import glob
import os
import cv2
import numpy as np
import json

global image_size
image_size = (227,227)

def calculate_bgr_channel_mean(root_directory,json_file_path):
        images_path = [y for x in os.walk(root_directory) for y in glob(os.path.join(x[0], "*.jpg"))]
        image = cv2.imread(images_path[0])[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = resize_in_aspect_to_ration(image)
        image = center_crop_image(image).astype(float)

        blue_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        red_channel = image[:, :, 2]

        red_mean = np.mean(red_channel)
        blue_mean = np.mean(blue_channel)
        green_mean = np.mean(green_channel)

        for i in range(len(images_path)):
            if i == 0:
                continue

            image = cv2.imread(images_path[i])[:, :, :3]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = resize_in_aspect_to_ration(image)
            image = center_crop_image(image).astype(float)
            blue_channel = image[:, :, 0]
            green_channel = image[:, :, 1]
            red_channel = image[:, :, 2]
            red_mean = (red_mean + np.mean(red_channel)) / 2
            blue_mean = (blue_mean + np.mean(blue_channel)) / 2
            green_mean = (green_mean + np.mean(green_channel)) / 2

        mean={}
        mean["blue_mean"] = blue_mean
        mean["green_mean"] = green_mean
        mean["red_mean"] = red_mean
        with open(json_file_path, 'w') as fp:
            json.dump(mean, fp)

    # Returns resized image in aspect to ratio
def resize_in_aspect_to_ration(image):
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