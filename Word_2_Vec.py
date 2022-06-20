# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 08:08:00 2022

@author: Eben Emmanuel
"""

## Import necessary Libraries
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec


## Create a corpus
speech = '''Fourscore and seven years ago our fathers brought forth, on this continent, a new nation, conceived 
in liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great 
civil war, testing whether that nation, or any nation so conceived, and so dedicated, can long endure. We are 
met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final 
resting-place for those who here gave their lives, that that nation might live. It is altogether fitting and 
proper that we should do this. But, in a larger sense, we cannot dedicate, we cannot consecrate—we cannot 
hallow—this ground. The brave men, living and dead, who struggled here, have consecrated it far above our poor 
power to add or detract. The world will little note, nor long remember what we say here, but it can never 
forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which 
they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great 
task remaining before us—that from these honored dead we take increased devotion to that cause for which they 
here gave the last full measure of devotion—that we here highly resolve that these dead shall not have died in 
vain—that this nation, under God, shall have a new birth of freedom, and that government of the people, by the 
people, for the people, shall not perish from the earth.'''


## Data cleaning and preprocessing
import re
text = re.sub(r'\[[0-9]*\]', ' ', speech)       #### Matching the given pattern
text = re.sub('\s+', ' ', text)             #### Remove the white-spaces
text = text.lower()
text = re.sub('\d', ' ', text)              #### Remove all digits
text = re.sub('\s+', ' ', text)


## Preparint the Data
sentences = nltk.sent_tokenize(text)            #### Sentence Tokenization
words = [nltk.word_tokenize(sentence) for sentence in sentences]        ##### Word Tokenization

for i in range(len(words)):
    words[i] = [word for word in words[i] if word not in stopwords.words('english')]
    

## Training Word2Vec Model
model = Word2Vec(words, min_count= 1)

vocabulary = model.wv.key_to_index    ### We obtain the Vocabulary for each key.
words_in_model = model.wv.index_to_key      ### This has the words or our keys.


## Finding Word Vectors
vectors = model.wv['fourscore']         ### Word2Vec creates a vector of 100 dimension


## Word Similarity
similar = model.wv.most_similar('work')

        #### Shows the similar based on this particular paragraph ####


