# ai-visual-storytelling-seq2seq
Implementation of our original solution that is described in the paper [Stories for Images-in-Sequence by using Visual and Narrative Components](https://arxiv.org/abs/1805.05622). Our project is inspired by the solution in [Visual Storytelling](https://arxiv.org/pdf/1604.03968.pdf).
The model generates stories, sentence by sentence with respect to the sequence of images and the previously generated sentence. The architecture of our solution consists of an image sequence encoder that models the sequential behaviour of the images, a previous-sentence encoder and a current-sentence decoder. The previous-sentence encoder encodes the sentence that was associated with the previous image and the current-sentence decoder is responsible for generating a sentence for the current image of the sequence. We also introduce a novel way of grouping the images of the sequence during the training process, in order to encapture the effect of the previous images in the sequence. Our goal with this approach was to create a model that will generate stories that contain more narrative and evaluative language and that every generated sentence in the story will be affected not only by the sequence of images but also by what has been previously generated in the story. 

### Installing
The project is built using Python 2.7.14, Tensorflow 1.6.0 and Keras 2.1.6. Install these dependencies to get a development env running
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
Download the Visual Storytelling Dataset (VIST) from http://visionandlanguage.net/VIST/dataset.html and save it in the dataset/vist_dataset directory. Also download the pre-trained [weights](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) for AlexnNet and put them in the dataset/models/alexnet directory. 

### Data pre-processing
First we need to extract the image features from all the images and save them in a file. This is possible with
```
python dataset/models/alexnet/myalexnet_forward_newtf.py
```
The script creates the file /dataset/models/alexnet/alexnet_image_train_features.hdf5, that contains all the image features.
Next we need to associate every image feature vector with it's corresponding vectorized sentence. We vectorize the sentence using the functions in sis_datareader. With the function sentences_to_index we align every image feature with every sentence. If all the file paths are set properly, all of the above can be done by running the command 
```
python data_reader/sis_datareader.py
```
### Options and differences from the paper
Other than our proposed solution, the project can be used to train an encoder-decoder and an encoder-decoder with [Luong](https://arxiv.org/pdf/1508.04025.pdf) attention mechanism.

### Our proposed solution
![alt text](https://github.com/Pendulibrium/ai-visual-storytelling-seq2seq/blob/master/architecture_horizontal.jpg)
The architecture of the proposed model. The images highlighted with red are the ones that are encoded and together with the previous sentence, they influence the generated sentence in the current time step.
### Training the model
Training the model and adjusting the parameters is done in the training_model.py. If the attention mechanism is used, make sure that image_encoder_latent_dim = sentence_encoder_latent_dim.
```
python training_model.py
```

### Generating stories
To generate stories in inference_model.py set model_name to the model you want to generate from and run
```
python inference_model.py
```
### Some Results
In following images we can see the generated stories from the aforementioned models. The first row reprsents the original story, the second row is the generated story from the model with loss=0.82, the third row is the story from the model with loss=1.01 and the forth row is the story from the model with loss=1.72. 
![alt text](https://github.com/Pendulibrium/ai-visual-storytelling-seq2seq/blob/master/results/images/slika1.png)
![alt text](https://github.com/Pendulibrium/ai-visual-storytelling-seq2seq/blob/master/results/images/slika2.png)
![alt text](https://github.com/Pendulibrium/ai-visual-storytelling-seq2seq/blob/master/results/images/slika3.png)
![alt text](https://github.com/Pendulibrium/ai-visual-storytelling-seq2seq/blob/master/results/images/slika4.png)
