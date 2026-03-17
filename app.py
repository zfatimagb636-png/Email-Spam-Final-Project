import streamlit as st
import pickle
import string

from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    # breaking into separate words
    text = nltk.word_tokenize(text)
    
    # as text is converted to list after tokenization- so using loop
    y=[]
    for i in text:
        if i.isalnum():   
         y.append(i)
    
    text=y[:]  
    y.clear()
    
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
            
    text=y[:]    
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.title("Email Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess (Ek Tab ka space)
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize (Ek Tab ka space)
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict (Ek Tab ka space)
    result = model.predict(vector_input)[0]
    
    # 4. Display (Ek Tab ka space)
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

# st.title("Email spam Classifier")
# input_sms= st.text_input("Enter the message")

# if  st.button('predict'):   
# #preprocessing
# transformed_sms = transform_text(input_sms) 

# #vectorize
# vector_input = tfidf.transform([transformed_sms])   

# #predict
# result = model.predict(vector_input)[0] 

# #display
# if result == 1: 
#    st.header("Spam")    
# else:   
#    st.header("Not Spam")    
