from flask import Flask
from flask import render_template
import sys

sys.path.insert(0, '../')
import h5py
from story_visualization import StoryPlot

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_folder='vist_dataset')


@app.route("/show_story/<int:story_index>")
def show_story(story_index):
    story_plot = StoryPlot(stories_data_set_path='./vist_sis/val.story-in-sequence.json',
                           images_root_folder_path='./vist_dataset/validate_data')

    train_dataset = h5py.File('./image_embeddings_to_sentence/stories_to_index_valid.hdf5', 'r')
    story_ids = train_dataset['story_ids']
    hypothesis = open('../results/2018-01-29_00:37:26-2018-01-31_00:01:42/hypotheses_valid.txt').read().split('\n')

    story_id = story_ids[story_index]
    hypotheses_sentences = hypothesis[(story_index * 5): (story_index * 5) + 5]
    data = story_plot.get_story_data(str(story_id))
    data['hypotheses_sentences'] = hypotheses_sentences

    print(data)

    return render_template('show_story.html', data=data)
