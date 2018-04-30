# ai-visual-storytelling-seq2seq
Implementation of seq2seq model for Visual Storytelling Challenge (VIST) http://visionandlanguage.net/VIST/index.html.
Our project is inspired by the solution in [Visual Storytelling](https://arxiv.org/pdf/1604.03968.pdf).
The model generates stories, sentence by sentence with respect to the sequence of images and the previously generated sentence. The architecture of our solution consists of an image sequence encoder that models the sequential behaviour of the images, a previous-sentence encoder and a current-sentence decoder. The previous-sentence encoder encodes the sentence that was associated with the previous image and the current-sentence decoder is responsible for generating a sentence for the current image of the sequence. We also introduce a novel way of grouping the images of the sequence during the training process, in order to encapture the effect of the previous images in the sequence. Our goal with this approach was to create a model that will generate stories that contain more narrative and evaluative language and that every generated sentence in the story will be affected not only by the sequence of images but also by what has been previously generated in the story. 

### Installing
The project is built using Python 2.7.14, Tensorflow 1.16.0 and Keras 2.1.6. Install these dependencies to get a development env running
```
sudo easy_install --upgrade pip
sudo easy_install --upgrade six
sudo pip install tensorflow
sudo pip install keras
pip install opencv-python
pip install h5py
pip install unidecode
python -mpip install matplotlib
```
### Data
Download the Visual Storytelling Dataset (VIST) from http://visionandlanguage.net/VIST/dataset.html.

### Data pre-processing

### Options and differences from the paper
Other than our proposed solution, the project can be used to train an encoder-decoder and an encoder-decoder with [Luong](https://arxiv.org/pdf/1508.04025.pdf) attention mechanism.

### Training the model

### Generating stories

### Some Results
