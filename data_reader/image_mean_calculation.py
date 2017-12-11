from glob import glob
import os
import cv2
import numpy as np
import json
import image_util as image_util

global image_size
image_size = (227,227)

def calculate_bgr_channel_mean(root_directory, json_file_path, image_size=(227,227)):
        images_path = [y for x in os.walk(root_directory) for y in glob(os.path.join(x[0], "*.jpg"))]
        images_path_png = [y for x in os.walk(root_directory) for y in glob(os.path.join(x[0], "*.png"))]
        images_path = np.append(images_path, images_path_png)
        image = cv2.imread(images_path[0])[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image_util.resize_in_aspect_to_ration(image=image, image_size=image_size)
        image = image_util.center_crop_image(image).astype(float)

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
            image = image_util.resize_in_aspect_to_ration(image=image, image_size=image_size)
            image = image_util.center_crop_image(image).astype(float)
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
