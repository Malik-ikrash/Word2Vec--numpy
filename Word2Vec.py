import numpy as np     #For matrix multiplication, dot product, etc used for all the linear algebra stuff
import re                            
import random
from collections import Counter


#Load and preprocess dataset
with open("dataset.txt", "r", encoding="utf8") as f:        
    text = f.read().lower()

# simple tokenization
tokens = re.findall(r"\b\w+\b", text)     #Turns text into words and re removes punctuallity 
tokens = tokens[:100000]         #Limits dataset size so its runs smoother and faster during training


#Build vocabulary
vocab_counts = Counter(tokens)     #Counts the number of times a word appears

min_count = 2
vocab = [w for w in vocab_counts if vocab_counts[w] >= min_count]      #Removes rare words that are less than the min_count because they are noise,
                                                                       # increases vocubalary , and slows down training 

word_to_idx = {w: i for i, w in enumerate(vocab)}           #Maps words to numbers for neural network
idx_to_word = {i: w for w, i in word_to_idx.items()}

vocab_size = len(vocab)

print("Vocabulary size:", vocab_size)

# Convert text to indices
data = [word_to_idx[w] for w in tokens if w in word_to_idx]      #Turns the text into their respective given number


#Negative sampling distribution
freq = np.array([vocab_counts[w] for w in vocab])               #Creates negative samples to select
neg_dist = freq ** 0.75                            #Multiply by 0.75 because it removes dominaice of extremely common words but still favours frequenct words
neg_dist = neg_dist / neg_dist.sum()


#Generate training pairs
window_size = 4       #Defines how many surrounding words count as context
pairs = []

for i, center in enumerate(data):

    for j in range(-window_size, window_size + 1):          #Checks the surrounding position and sees if they can pair up 

        if j == 0:
            continue

        context_pos = i + j

        if context_pos < 0 or context_pos >= len(data):
            continue

        context = data[context_pos]

        pairs.append((center, context))

print("Training pairs:", len(pairs))


#Model parameters
embedding_dim = 150     #Dimensions of each word
learning_rate = 0.004  #Learning rate of the model
epochs = 8     #number of times the dataset is seen by the model
negative_samples = 5     #number of times incorrect samples are given during training to help the model learn

# Input embeddings
W_in = np.random.randn(vocab_size, embedding_dim) * 0.01     #A random input embedding so the model learns

# Output embeddings
W_out = np.random.randn(vocab_size, embedding_dim) * 0.01    #A random output embedding so the model learns


#Helper functions
def sigmoid(x):
    x = np.clip(x , -10, 10)
    return 1 / (1 + np.exp(-x))      #Turns it into a probability, the bigger the probavility the likely chance the word is correct


def sample_negative(k, context):
    negatives = []
    while len(negatives) < k:
        n = np.random.choice(vocab_size, p=neg_dist)
        if n != context:   #Ensures we do not accidentally sample the real context word
            negatives.append(n)
    return negatives     #Selects random incorrect words so the model learns from it 



#Training loop
for epoch in range(epochs):     #loops through

    total_loss = 0

    random.shuffle(pairs)     #shuffles so the model doesn't learn order patterns

    #Learning rate decay so updates get smaller over time
    lr = learning_rate * (1 - epoch / epochs)

    for center, context in pairs:

        v_c = W_in[center]
        v_o = W_out[context]

        #Create gradient accumulators so we don't update using stale vectors
        grad_center = np.zeros_like(v_c)
        grad_out_context = np.zeros_like(v_o)

        #Forward pass 
        score_pos = sigmoid(np.dot(v_c, v_o))     #Dot product between embeddings. A high dot product meant similar context

        #Loss 
        loss_pos = -np.log(score_pos + 1e-10)     #Loss measures prediction error. So should be it low as possible 
        total_loss += loss_pos

        #Gradient
        grad_pos = score_pos - 1           #Used to see the models prediction. Low gradient means the model predicts words correctly

        #Accumulate gradients instead of updating immediately
        grad_center += grad_pos * v_o
        grad_out_context += grad_pos * v_c

        #Negative sampling
        negatives = sample_negative(negative_samples, context)          # Sample random words that are NOT the correct context word

        for neg in negatives:

            v_n = W_out[neg]                  # Get embedding of the negative word

            score_neg = sigmoid(np.dot(v_c, v_n))           # Compute probability that the negative word appears near the center word

            loss_neg = -np.log(1 - score_neg + 1e-10)       # Loss for negative samples (we want this probability to be close to 0)
            total_loss += loss_neg

            grad_neg = score_neg        # Gradient for the negative sample

            grad_center += grad_neg * v_n             #Accumulate gradient for center word
            W_out[neg] -= lr * grad_neg * v_c             # Update embeddings using stochastic gradient descent

        #Apply updates after gradients are accumulated
        W_in[center] -= lr * grad_center       #Adjust the center word embedding so that it reduces the prediction error (loss)
        W_out[context] -= lr * grad_out_context     #Adjust the context word embedding so that it reduces the prediction error (loss)

    print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}")       # Print total loss after each training epoch

# Normalize embeddings after training
W_in = W_in / np.linalg.norm(W_in, axis=1, keepdims=True)


#Word similarity
def most_similar(word, top_n=5):            #Loop cosine similarity

    if word not in word_to_idx:            #Checks if the word exists in the vocabulary
        print("Word not in vocabulary")
        return []

    idx = word_to_idx[word]             #Gets the embedding vector of the input word
    vec = W_in[idx]

    similarities = []

    for i in range(vocab_size):        #Compares the word with every other word in the vocabulary

        if i == idx:
            continue

        other = W_in[i]             #Get embedding vector of another word

        sim = np.dot(vec, other) / (                        #Compute cosine similarity between word vectors
            np.linalg.norm(vec) * np.linalg.norm(other)
        )

        similarities.append((idx_to_word[i], sim))      #Store the word and its similarity score

    similarities.sort(key=lambda x: x[1], reverse=True)       #Sorts the word by their similarity score

    return similarities[:top_n]             #Returns the top n most similar words


#Word analogy function
def analogy(word_a, word_b, word_c, top_n=5):

    # Check if words exist in vocabulary
    if word_a not in word_to_idx or word_b not in word_to_idx or word_c not in word_to_idx:
        print("One of the words is not in vocabulary")
        return []

    # Get embeddings
    vec_a = W_in[word_to_idx[word_a]]
    vec_b = W_in[word_to_idx[word_b]]
    vec_c = W_in[word_to_idx[word_c]]

    # Perform vector arithmetic
    target_vec = vec_a - vec_b + vec_c

    similarities = []

    for i in range(vocab_size):

        word = idx_to_word[i]

        if word in [word_a, word_b, word_c]:
            continue

        other_vec = W_in[i]

        sim = np.dot(target_vec, other_vec) / (
            np.linalg.norm(target_vec) * np.linalg.norm(other_vec)
        )

        similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]

#Example usage
print("\nSimilar words to 'love':")
print(most_similar("love"))

print("\nAnalogy test: father - man + woman")
print(analogy("father", "man", "woman"))