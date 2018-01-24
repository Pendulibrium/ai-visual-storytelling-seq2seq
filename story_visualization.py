import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from glob import glob
from PIL import Image
import cv2
import textwrap
import os


class StoryPlot:
    def __init__(self, stories_data_set_path='./dataset/vist_dataset/validate_data/val.story-in-sequence.json',
                 images_root_folder_path='./dataset/vist_dataset/validate_data/images/val'):

        self.story_dataset_path = stories_data_set_path
        self.images_root_folder_path = images_root_folder_path
        self.annotations = json.load(open(stories_data_set_path))['annotations']

    def visualize_story(self, story_id, decoded_sentences):

        story = []
        for annotation_data in self.annotations:
            annotation = annotation_data[0]
            if annotation['story_id'] == story_id:
                story.append(annotation)

        story = sorted(story, key=lambda k: k['worker_arranged_photo_order'])
        story_image_filenames = [''] * len(story)
        image_paths = [y for x in os.walk(self.images_root_folder_path) for y in glob(os.path.join(x[0], "*.jpg"))]
        for filename in image_paths:
            for i in range(len(story)):
                if story[i]['photo_flickr_id'] in filename:
                    story_image_filenames[i] = filename

        fig = plt.figure()

        wrapper = textwrap.TextWrapper(width=40)
        for i in range(len(story_image_filenames)):
            im = cv2.imread(story_image_filenames[i])
            im = cv2.resize(im, (300, 300))

            original_text = story[i]['text']
            decoded_text = decoded_sentences[i]

            a = fig.add_subplot(1, len(story_image_filenames), i + 1)

            a.axis("off")
            a.text(0, 330, "\n".join(wrapper.wrap(original_text)), ha='left', va="top")
            a.text(0, 400, "\n".join(wrapper.wrap(decoded_text)), ha='left', va="top")

            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        plt.axis("off")
        plt.show()

# story_plot = StoryPlot(stories_data_set_path='./dataset/vist_sis/train.story-in-sequence.json',
#                        images_root_folder_path='./dataset/sample_images')
# story_plot.visualize_story("11053", [
#     "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard",
#     "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard",
#     "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard",
#     "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard",
#     "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard"])
