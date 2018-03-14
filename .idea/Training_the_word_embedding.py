#importing the import module
from gensim.models import Word2Vec

#The sentence in the list is

sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'], ['this', 'is', 'the', 'second', 'sentence'], ['yet', 'another', 'sentence'], ['one', 'more', 'sentence'], ['and', 'the', 'final', 'sentence']]

model=Word2Vec(sentences,min_count=1)

#printing the model

print("this is the trained model", model)

#printing the model

words=list(model.wv.vocab)

print("printing the vocab",words)

#printing the vector

print("printing the vector")
print(model["another"])


