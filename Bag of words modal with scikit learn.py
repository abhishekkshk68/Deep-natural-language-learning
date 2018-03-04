from sklearn.feature_extraction.text import TfidfVectorizer

list_of_doc = ["The quick brown fox jumped over the lazy dog.",
"The dog.",
"The fox"]

#create  the transofrmer
vectorizer = TfidfVectorizer()


#tokenisze and build the vovab
k=vectorizer.fit(list_of_doc)

print("~~~~~~~printing the vector~~~~~~~~~~~")
# printing the output
print(vectorizer.vocabulary_)

print("~~~~~~~printing the idf of the vector~~~~~~~~~~~")

print(vectorizer.idf_)
# encode document

print("~~~~~~~printing the encoder of the document~~~~~~~~~~~")

vector = vectorizer.transform([list_of_doc[0]])

print(vector)
# summarize encoded vector

print("~~~~~~~printing the shape of the vector~~~~~~~~~~~")
print(vector.shape)

print("~~~~~~~printing the aarray~~~~~~~~~~~")
print(vector.toarray())






