# Image caption_generator: using CNN and LSTM models
To generate a caption for any image in English language. The module is built using  [keras](https://keras.io/), the deep learning library. 

## Datasets used:
- ## Flickr8k_text
  - __Flickr_8k.trainImages.txt__  
    - contains all the train_image names (for example - '218342358_1755a9cce1.jpg')
  - __Flickr_8k.testImages.txt__
    - contains all the test_image names (for example - '218342358_1755a9cce1.jpg') 
  - __Flickr8k.token.txt__
    - contains all the raw captions of the Flickr8k Dataset. The first column is the
    ID of the caption which is "image address # caption number"
- ## Flicker8k_Dataset
  - Contains all the raw images in the dataset - 8091 images     

## Prepared Datasets/textfiles
- `descriptions.txt` - this is the text file that has a key-value pair of each image with its 5 captions 
- `tokenizer.p` - this is a _'pickle'_ file that contains all the words, fit on the tokenizer.
- `features.p` - This is the _'pickle'_ file, that contains all the features of the images, obtained by using
    the pretrained model. - __Xception__


## Libraries used

- __tensorflow.keras.utils.plot_model()__ - to plot the model
- __keras.preprocessing.text.Tokenizer__ - Using a  tokenizer for creating a vocabulary
- __numpy__
- __PIL.Image__
- __os library__
- __pickle.dump, pickle.load__ - to load and unload data to and from _'pickle'_ file
- __Xception__ - pretrained model to extract image features from the images
- __load_img, img_to_array__ - from keras, to load images
- __keras.preprocessing.sequence__ - importing pad_sequences for text data
- __tensorflow.keras.utils.to_categorical__ - to convert data to categorical data
- __keras.models.Model, keras.models.load_model__ - Model loading parameters
- __tensorflow.keras.layersDense, LSTM, Embedding, Dropout__ - different layers for the model
- __bleu scores - corpus_bleu,sentence_bleu,SmoothingFunction(from nltk)__ - for
calculating BLEU score



## Model 

The Image captioning model has been implemented using the Sequential API of keras. It consists of three components:

- __[Xception](https://keras.io/api/applications/xception/)__ - Pretrained Image CNN feature extraction model(CNN with 71 layers depth), that will extract features from the images in the 'FlickerDataset'

- __[LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)__ - A Recurrent neural neural network that is used in creating encoder-decoder models
- __Sequence_Processor__
    -__[Embedding](https://keras.io/api/layers/core_layers/embedding/)__ - The Embedding Layer will handle the textual input sent to the model
- __Feature Extractor:__ Features of image(each image feature of size 2048) from the 
    Xception model, are converted to 256 size   
- __Decoder:__ By merging the 2 layers from the encoder(features and embedded text), the decoder will refine data through a dense layer.
## Requirements 


| Requirements | versions(if any) |
| ------ | ------ |
| __Pillow__ |  |
| __System Specific Parameters(sys package)__ | version 3.8.11 |
| __numpy__ | version 1.20.3 |
| __Tensorflow__ | version 2.8.0 |
| __pandas__ | version 1.3.2 |

These package can be easily installed by:
    `!pip install package_name`


## Code Walkthrough


### Scripts

- ### caption_preprocessing -1.py 
  This script is used to run various functions, that preprocess the captions of the images and finally output a file that contains a __key-value pair__ of an image     and their respective captions(5 captions for each image) for training.
 
	They have the following functions
	- __`load_docs(file)`__ - it loads the text file for reading
	- __`all_img_captions(file)`__ - this function creates a dictionary of images and their respective captions
	- __`clean_text(img_data)`__ -this function removes all the unecessary symbols and fixes casing of all the captions in the dictionary
	- __`word_vocab(desc)`__ - this function just gives the vocabulary of the text in captions(just for checking, not relevant)
	- __`descriptions_txt(desc,file)`__ - this function saves the descriptions into a text tile 'file'



 
- ### feature_extraction.py 
  Feature Extraction script is used to extract image features of images in the 'Flicker8KDataset', using the pretrained CNN model __Xception__
  function in this script is below:

	- __`image_features_extraction(directory)`__ - This will extract features for all images will map the image_names with their respective feature array.
	- These features are then dumped into a `features.p` _pickle_ file.





- ### model_train.py 
  after generating all the parameters for the model, this python function creates the model and trains the model with the features from images and the captions for     that image

	The __model_train.py__ file is run using the following functions:
	- __`load_docs(filename)`__:- Loads the file
	- __`all_img_captions(file)`__:- Gets dictionary of imagename mapped with its respective captions
	- __`photo_data(file)`__:- Loads the photos from the dataset 
	- __`load_clean_descriptions(file, photos)`__:- performs cleaning and preprocessing on captions
	- __`photo_features(photos)`__ :- this function loads the features we extracted from the photos in dataset, using the Xception Model. These features were 	                                    stored in a _pickle_ file __'features.p'__
	- __`create_tokenizer(descriptions)`__:-  Tokenizer will map every word in the text corpus with a unique index
	- __`maximum_length(descriptions)`__ a function to get the maximum length of a caption in description
	- __`create_sequences(tokenizer, max_length, desc_list, feature)`__:- creates sequence of x1(feature), sequence x2(input_seq word) and output y (output word)
	- __`data_generator(desc, features, tokenizer, max_length)`__:- this function is used by the model.fit_generator() to load data
	- __`define_model(vocab_size, max_length)`__:- this function creates and defines the model 
    
- ### model_test.py 
  once the model is trained, we can run test the model using this script.
	functions used
	- __`load_docs(filename)`__
	- __`photo_data(file)`__
	- __`extract_features_test(filename, model)`__
	- __`word_for_id(integer, tokenizer)`__ :- finding if word exists for a paticular index value
	- __`generate_desc_test(model, tokenizer, photo, max_length)`__ :- generatest a caption for test data

### Usage
After executing the required packages, below scripts are to be executed in the following order
1. #### caption_preprocessing -1.py
2. #### feature_extraction.py
3. #### model_train.py
4. #### model_test.py

