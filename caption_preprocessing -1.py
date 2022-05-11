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


def load_docs(filename):
    #opening the file in read format
    f = open(filename, 'r')
    text = f.read()
    f.close()
    return text

#getting a dictionary of all the images and their captions
def all_img_captions(file):
    desc = {}
    file = load_docs(file)
    captions = file.split('\n')

    for cap in captions[:-1]:
        img, caption = cap.split('\t')
        #appending the captions for a paticular image from file
        #key->img ; value->caption
        if img[:-2] not in desc:
            desc[img[:-2]] = [ caption ]
        else:
            desc[img[:-2]].append(caption)

    #returning the dictionary
    return desc

#Cleaning the caption- removing punctuations and words containing numbers,changing words to lower case
def clean_text(img_data):
    table = str.maketrans('','',string.punctuation)
    for img,cap in img_data.items():
        for i,img_cap in enumerate(cap):
            img_cap.replace("-"," ")
            sentence = img_cap.split()
            #converts to lowercase
            sentence = [a.lower() for a in sentence]
            #remove punctuation from each word in caption
            sentence = [a.translate(table) for a in sentence]
            #remove words like 's and a
            sentence = [a for a in sentence if(len(a)>1)]
            #remove tokens with numbers in them
            sentence = [a for a in sentence if(a.isalpha())]
            #convert list of words, back to string
            img_cap = ' '.join(sentence)
            img_data[img][i]= img_cap
    return img_data



#  making a vocabulary of all the unique words
def word_vocab(desc):
    voc = set()
    for key in desc.keys():
        [voc.update(d.split()) for d in desc[key]]
    return voc

#Saving all descriptions in a file
def descriptions_txt(desc, file):
    lines = list()
    for key, desc_list in desc.items():
        for d in desc_list:
            lines.append(key + '\t' + d )
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()








#directory of all the captions and image name data
dataset_text = "Flickr8k_text"

#directory of all the images
dataset_images = "Flicker8k_Dataset"


#making the directory for caption and image-text data
filename = dataset_text + "/" + "Flickr8k.token.txt"



#mapping the images with their respective captions(each image has 5 captions)
descriptions = all_img_captions(filename)
print("Length of descriptions =" ,len(descriptions))



#cleaning the captions in description dict
clean_descriptions = clean_text(descriptions)

#getting the vocabulary of words
vocabulary = word_vocab(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))


#saving each description to file- descriptions.txt
descriptions_txt(clean_descriptions, "descriptions.txt")
