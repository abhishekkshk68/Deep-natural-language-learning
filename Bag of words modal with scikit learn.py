from sklearn.feature_extraction.text import TfidfVectorizer

list_of_doc = ["The quick brown fox jumped over the lazy dog.",
"The dog.",
"The fox"]

#create  the transofrmer
vectorizer = TfidfVectorizer()

#tokenisze and build the vovab
k=vectorizer.fit(list_of_doc)

# printing the output
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([list_of_doc[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())




