import streamlit as st 
import numpy as np 
import pandas as pd 
from sklearn import datasets
import base64
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="IRIS Prediction",page_icon="image/icon.png",layout="wide",initial_sidebar_state="collapsed")

#app header
cols = st.beta_columns(3)
with cols[0]:
    st.markdown(
            f"""
            <div class="container">
                <img class="logo-img" style="width:100px" src="data:image/png;base64,{base64.b64encode(open("./image/iris.jpeg", "rb").read()).decode()}">
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("<h1 style=' color:#9f73f7'>Iris Flowers</h1>", unsafe_allow_html=True)

st.write("""
# Simple Iris Flower Prediction App
This App predicts the  IRIS Flowers type 
""")

st.sidebar.header("User Input Parameters")
#the variation of this parameters determine the type of iris flowers
def user_input_features():
    sepal_length = st.sidebar.slider("sepal_length", 4.3,7.9,5.4)
    sepal_width = st.sidebar.slider("sepal_width", 2.0,4.4,3.4)
    petal_length = st.sidebar.slider("petal_length", 1.0,6.9,1.3)
    petal_width = st.sidebar.slider("petal_width", 0.1,2.5,0.2)

    data = {'sepal_length':sepal_length,
    'sepal_width':sepal_width,
    'petal_length':petal_length,
    'petal_width':petal_width}
    features = pd.DataFrame([data], index=[0])
    return features

df = user_input_features()

st.subheader("User Input")
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X,Y)

predictions = clf.predict(df)
predict_proba = clf.predict_proba(df)

st.subheader("Class Labels and their corresponding index number")
st.write(iris.target_names)

st.subheader("Prediction")
st.write(iris.target_names[predictions])

st.subheader("Prediction probability")
st.write(predict_proba)