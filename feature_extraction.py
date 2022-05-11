import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TF to use only the CPU
import sys
import numpy as np
import tensorflow as tf
import pandas as pd

import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
# small library for seeing the progress of loops.
from tqdm import tqdm_notebook as tqdm
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu,SmoothingFunction
import matplotlib.pyplot as plt




def image_features_extraction(directory):
    model = Xception(include_top=False, pooling='avg')
    features={}
    #each image - i
    for i in tqdm(os.listdir(directory)):
        file = directory + "/" + i
        img = Image.open(file)
        img = img.resize((299,299))
        img = np.expand_dims(img, axis=0)
        #image = preprocess_input(image)
        img = img/127.5
        img = img - 1.0
        feature = model.predict(img)
        features[i] = feature
    return features



#2048 feature vector
dataset_images = "Flicker8k_Dataset"
features = image_features_extraction(dataset_images)
dump(features, open("features.p","wb"))
