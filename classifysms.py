import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
nltk.download('punkt_tab')
nltk.download('stopwords')

model = pickle.load(open('modelsms.pkl', 'rb'))
tf = pickle.load(open('tf.pkl', 'rb'))

st.title("Email/ SMS classifier")
text = st.text_area("Enter the message")

# given an input we have to do following steps
# text transform 
def transform_text(text):
  text = text.lower()
  # tokenization
  text = nltk.word_tokenize(text)

  y = []
  # removing special characters
  for i in text:
    if i.isalnum():
      y.append(i)

  # remove stop words
  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  # stemming
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

if st.button("Predict"):
    data = transform_text(text)
    # vectorize 
    vector = tf.transform([data])
    # give it to our model 
    result = model.predict(vector)[0]
    # check for if 1 print spam if 0 print not spam
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")