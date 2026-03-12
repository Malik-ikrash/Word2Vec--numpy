# Word2Vec--numpy
A Word2Vec skip-gram model with negative sampling implemented from scratch in Python using NumPy. Learn word embeddings, compute word similarity, and perform analogy tests without using any ML frameworks.

---

## Features in the program:

- Skip-gram Word2Vec architecture  
- Negative sampling for efficient training  
- Stochastic Gradient Descent (SGD) optimization  
- Cosine similarity for finding similar words  
- Word analogies (e.g., king - man + woman ≈ queen)  

---

## Dataset:

This project uses a small text from Shakespeare (Roughly 1 millioin words).  
To make the training model faster, I limited the text to 100000 tokens. 
if you wish to increase the number of tokens the program learns on then change this bit of the code:
    tokens = tokens[:100000]
The number inside the close brackets and after the colon controls the number of words the program learns on
change it to whether you like, though keep in mind that it doesn't excede the dataset.txt.
If you wish to use the whole txt file, then simply remove the above code all together and the program 
will use the whole txt file.

> Note: The quality of word similarity and analogies improves with larger datasets/when you use more tokens (words).

---

## Installing the code:

Clone the repository in terminal and change to folder (two commands):

    git clone https://github.com/Malik-ikrash/word2vec--numpy.git
    cd word2vec--numpy

## Create a virtual python environment (two commands):
    python3 -m venv venv
    source venv/bin/activate

## Install dependencies:
    pip install -r requirements.txt

## Type/copy this into terminal for the program to run: 
    python Word2Vec.py

## Output should look something like:

    Vocabulary size: 2566
    Training pairs: 377660
    Epoch 1/8 Loss: 1459184.3387
    ...
    Similar words to 'love': heart, mother, god, father
    Analogy test: king - man + woman ≈ liege

  > Note: Similarity depends on how many tokens you use. The more tokens you use, the better the results will be.

  







