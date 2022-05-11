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


#load the data
def photo_data(file):
    f = load_docs(file)
    photos = f.split("\n")[:-1]
    return photos

def extract_features_test(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature



def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None



def generate_desc_test(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

dataset_text = "Flickr8k_text"
test_filename = dataset_text + "/" + "Flickr_8k.testImages.txt"
test_imgs = load_photos(test_filename)
test_description = load_clean_descriptions("descriptions.txt", test_imgs)


#path = 'Flicker8k_Dataset/111537222_07e56d5a30.jpg'
max_length = 32

tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
i=1

#predicting the results of first 30 pictures
for img in test_imgs[:30]:
    img_path = 'Flicker8k_Dataset' + "/" + img
    photo = extract_features_test(img_path, xception_model)

    references = np.array(test_description[img])



    img = Image.open(img_path)
    description = generate_desc_test(model, tokenizer, photo, max_length)
    # print("\n\n")
    # print(description)
    plt.title("Picture"+str(i),fontsize=20)
    plt.imshow(img)
    plt.show()




    #print(references[0][8:-6])
    #print(references[1][8:-6])
    #print(references[2][8:-6])
    #print(references[3][8:-6])
    #print(references[4][8:-6])
    x_pr = description[5:-3].split()
    x_tr = [references[0][8:-6].split(),references[1][8:-6].split(),references[2][8:-6].split(),references[3][8:-6].split(),references[4][8:-6].split()]




    print("\n\nActual Sentences:\n")
    print(references[0][8:-6])
    print(references[1][8:-6])
    print(references[2][8:-6])
    print(references[3][8:-6])
    print(references[4][8:-6])

    print('\n\nPredicted Sentence:\n',description[5:-3])
    #score = corpus_bleu(references, candidates)
    #print(score)

    #print(x_tr)
    #print(x_pr)
    chencherry = SmoothingFunction()
    score = sentence_bleu(x_tr, x_pr,smoothing_function=chencherry.method4)
    print('Cumulative 1-gram: %f' % sentence_bleu(x_tr, x_pr, weights=(1, 0, 0, 0)))
    print('Cumulative 2-gram: %f' % sentence_bleu(x_tr, x_pr, weights=(0.5, 0.5, 0, 0)))
    print('Cumulative 3-gram: %f' % sentence_bleu(x_tr, x_pr, weights=(0.33, 0.33, 0.33, 0)))
    print('Cumulative 4-gram: %f' % sentence_bleu(x_tr, x_pr, weights=(0.25, 0.25, 0.25, 0.25)))
    print("BLEU SCORE using smoothing functions",score)
    i+=1
