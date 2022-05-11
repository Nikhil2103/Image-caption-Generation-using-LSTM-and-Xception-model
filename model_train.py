import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TF to use only the CPU
import sys
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
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





#load the data
def photo_data(file):
    f = load_docs(file)
    photos = f.split("\n")[:-1]
    return photos


def load_clean_descriptions(file, photos):
    #loading clean_descriptions
    desc = {}
    f = load_docs(file)

    for line in f.split("\n"):
        w = line.split()
        if len(w)<1 :
            continue
        img, caption = w[0], w[1:]
        if img in photos:
            if img not in desc:
                desc[img] = []
            sentence = '<start> ' + " ".join(caption) + ' <end>'
            desc[img].append(sentence)
    return desc


def photo_features(photos):
    #loading all features
    photo_fts = load(open("features.p","rb"))
    #selecting only needed features
    fts = {k:photo_fts[k] for k in photos}
    return fts



#creating tokenizer class
#this will vectorise text corpus
#each integer will represent token in dictionary
def create_tokenizer(descriptions):
    desc=[]
    for key in descriptions.keys():
        [desc.append(d) for d in descriptions[key]]

    #desc_list = dict_to_list(descriptions)
    t = Tokenizer()
    t.fit_on_texts(desc)
    return t





#calculate maximum length of descriptions
def maximum_length(descriptions):
    desc=[]
    for key in descriptions.keys():
        [desc.append(sentence) for sentence in descriptions[key]]

    max_length = max(len(sentence.split()) for sentence in desc)
    return max_length


def create_sequences(tokenizer, max_length, desc_list, feature):
    X1 = list()
    X2 = list()
    y  = list()

    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)




#create input-output sequence pairs from the image description.
#data generator, used by model.fit_generator()
def data_generator(desc, features, tokenizer, max_length):
    while 1:
        for key, desc_list in desc.items():
            #getting the photo features
            feature = features[key][0]
            inp_image, inp_sequence, out_word = create_sequences(tokenizer, max_length, desc_list, feature)
            yield [[inp_image, inp_sequence], out_word]







def define_model(vocab_size, max_length):
    # features from the CNN model squeezed from 2048 to 256 nodes
    i1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(i1)
    fe2 = Dense(256, activation='relu')(fe1)
    # LSTM sequence model
    i2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(i2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[i1, i2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model





#main


dataset_text = "Flickr8k_text"
dataset_images = "Flicker8k_Dataset"
token_file = dataset_text + "/" + "Flickr8k.token.txt"


descriptions = all_img_captions(token_file)
features = load(open("features.p","rb"))
filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"
#train = loading_data(filename)
train_imgs = photo_data(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = photo_features(train_imgs)






# each word gets an index, and they are stored into the tokenizer.p file
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print("vocabulary length",vocab_size)

max_length = maximum_length(descriptions)
print("maximum length of a caption",max_length)



# model training using data-generator
print('Original Dataset lenght: ', len(train_imgs))
print('Length of train data=', len(train_descriptions))
print('Length of train features=', len(train_features))
print('Vocabulary Length:', vocab_size)
print('Description Length: ', max_length)
model = define_model(vocab_size, max_length)
epochs = 10
#steps -data_iter
data_iter = len(train_descriptions)
# directory to save models
#os.mkdir("models")

for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit_generator(generator, epochs=1, steps_per_epoch= data_iter, verbose=1)
    model.save("models/model_" + str(i) + ".h5")
