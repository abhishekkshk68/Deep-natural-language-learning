from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["Indian Heritage Hotels Association· Federation of Hotels and Restaurant Associations of India · Hotel Association of India. GST Facilitation Centre of Ministry of Tourism: Email- gstontourism@gmail.com / Call- 011- 23793846 GST Facilitation Centre of Ministry of Tourism: Email- gstontourism@gmail.com / Call- 011- ...",
          "Incredible!ndia (@incredibleindia) · Twitter https://twitter.com/incredibleindia Tsomoriri Wetland Conservation Reserve in Ladakh is home to around 14 species of water birds including vulnerable species such as Black-necked Cranes, Bar-headed Geese, Ferruginous Pochard and Black-necked Grebe. #Wetland #IncredibleIndia @alphonstourism @tourismgoi pic.twitter.com/L5K74Q9… 2 hours ago · Twitter The Nalia Grasslands in Gujarat is a breeding ground for the endangered Great Indian Bustard. #IncredibleIndia 📸nomadandabag.blogspot.i… pic.twitter.com/wSdhmVL…3 hours ago · Twitter Fly high like a bird. Go paragliding in Solang Valley; get ready for an exhilarating, adrenaline-pumping experience! #FridayFeeling #IncredibleIndia pic.twitter.com/uNBFavA… 5 hours ago · Twitter",
          "India Tourism: TripAdvisor has 5735852 reviews of India Hotels, Attractions, and Restaurants making it your best India resource."]
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
idf = vectorizer._tfidf.idf_
print (dict(zip(vectorizer.get_feature_names(), idf)))