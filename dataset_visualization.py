from story_visualization import StoryPlot
import h5py

story_plot = StoryPlot(stories_data_set_path='./dataset/vist_sis/train.story-in-sequence.json',
                       images_root_folder_path='./dataset/sample_images')

train_dataset = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_train.hdf5', 'r')
story_ids = train_dataset['story_ids']
hypothesis = open('./results/hypotheses_2018-01-18_17:39:24-2018-01-20_18:50:39_train.txt').read().split('\n')

story_index = 0
story_id = story_ids[story_index]

while True:

    story_plot.visualize_story(story_id, hypothesis[story_index: story_index + 5])

    direction = raw_input('Prev/Next q-w:')
    if direction == 'q':
        story_index = story_index - 1
    elif direction == 'w':
        story_index = story_index + 1
