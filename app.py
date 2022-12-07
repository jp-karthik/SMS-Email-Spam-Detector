import sklearn
import streamlit as st
import pickle as pk
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

tfidf = pk.load(open('vectorizer.pkl', 'rb'))
model = pk.load(open('model.pkl', 'rb'))

# st.set_page_config(layout="wide")
ps = PorterStemmer()

def perform_data_pre_processing(text) : 
    text = text.lower() # converto lower case
    text = nltk.word_tokenize(text) # tokenization
    
    # retaining alpha numeric words
    y = []
    for word in text : 
        if word.isalnum() :
            y.append(word)
  
    text.clear()
    text = y[:] # cloning y into text
    y.clear()
        
    # removing stop words and punctuations
    for word in text : 
        if word not in stopwords.words('english') and word not in string.punctuation : 
            y.append(word)
    
    text.clear()
    text = y[:] # cloning y into text
    y.clear()
    
    #stemming
    for word in text : 
        y.append(ps.stem(word))
        
    text = " ".join(y)
    
    return text

# interface code

st.title("Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict') : 

    # preprocess text
    transformed_text = perform_data_pre_processing(input_sms)

    # vectorize tranformed text
    vectorized_input = tfidf.transform([transformed_text])
    
    # store model prediction in result
    result = model.predict(vectorized_input)

    if result[0] == 1 : 
        st.error("Spam")
    else :
        st.success("NotSpam")
