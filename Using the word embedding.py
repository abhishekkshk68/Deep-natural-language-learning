from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
             ['this', 'is', 'the', 'second', 'sentence'],
             ['yet', 'another', 'sentence'],
             ['one', 'more', 'sentence'],
             ['and', 'the', 'final', 'sentence']]

model=Word2Vec(sentences,min_count=1)

#taking the vectors

X=model[model.wv.vocab]

#printing the vocab

#print(model['more'])

#fitting the vector to the pca model
pca=PCA(n_components=2)

result=pca.fit_transform(X)

print("printing the results")
print(result.shape)

print(result[:,0])

print("the results are")

print(result)
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

pyplot.show()
