import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from glob import glob
from PIL import Image
import cv2

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

        for filename in glob(self.images_root_folder_path + '/*.jpg'):
            for i in range(len(story)):
                if story[i]['photo_flickr_id'] in filename:
                    story_image_filenames[i] = filename

        fig = plt.figure()
        print(story_image_filenames)
        for i in range(len(story_image_filenames)):
            im = cv2.imread(story_image_filenames[i])
            im = cv2.resize(im, (227, 227))
            original_text = story[i]['text']
	    original_text = original_text[:40] + "\n" + original_text[40:]
            decoded_text = decoded_sentences[i][:40] + "\n" + decoded_sentences[i][40:]
	    a = fig.add_subplot(1, len(story_image_filenames), i + 1)
            a.axis("off")
            a.text(0, 250, original_text, ha = 'left', wrap = True)
            a.text(0, 290, decoded_text, ha = 'left', wrap = True)

            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


        plt.axis("off")
        plt.show()

