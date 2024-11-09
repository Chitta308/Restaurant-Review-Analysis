# frontend
import streamlit as st 
import pickle

import re
import nltk
from nltk.corpus import stopwords # for stopwords
from nltk.stem.porter import PorterStemmer # for stem the words


model=pickle.load(open(r"C:\Users\chitt\VS Code Projects\Resturant_Review Analysis\model.pkl","rb"))
tfidf=pickle.load(open(r"C:\Users\chitt\VS Code Projects\Resturant_Review Analysis\tfidf.pkl",'rb'))

st.title("Restaurant Review Analysis ")

st.write("""
### About the App
This *Feedback Analysis App* uses Natural Language Processing (NLP) to analyze customer reviews and predict their sentiment. Just enter any feedback or review, and the app will determine whether the sentiment is positive or negative. This can be helpful for businesses to quickly suggestion customer satisfaction and improve service quality.
""")

inputText=st.text_area("Enter your review Here")

corpus=[]


# take to proper format
for i in inputText:
    review = re.sub('[^a-zA-Z]', ' ', inputText)
    review = review.lower()
    review = review.split()
    ps=PorterStemmer()   
    review = ' '.join(review)
    corpus.append(review)
    
# Transform the text using the tfidf
    review_vector = tfidf.transform([review])  # Transform to the tfidf format

if st.button("Click Here"):
    predict=model.predict(review_vector)
    if predict==1:
        st.success(f"Your feed back is positive review. ")
    else:
        st.success("Your feed back in negative.")
