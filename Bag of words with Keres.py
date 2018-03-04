from keras.preprocessing.text import Tokenizer
# define 5 documents
docs = ['Well done! well',
'Good work',
'Great effort',
'nice work well',
'Excellent!']
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
print("the word we leann")
# summarize what was learned
print("The total words count")
print(t.word_counts)
print("The word counts are")
print(t.document_count)
print("the word index is")
print(t.word_index)
print("the word documents are")
print(t.word_docs)
# integer encode documents

print("encoded documents")
encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)