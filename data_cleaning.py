from nltk.tokenize import word_tokenize

input_file="Guten_berg_text_file_input.txt"
out_put_file="Cleaned_data.txt"
with open (input_file,'r+',errors='ignore') as f:

  text_data=f.read()


tokens=word_tokenize(text_data)

with open(out_put_file,"w",errors='ignore') as k:
  for token in tokens:  
      k.write(token)


print ("The Cleaning operations are completed")
