import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

#loading files
df=pd.read_csv("data.csv",encoding='latin-1')

#fix missing values
df['clean_text']=df['clean_text'].fillna("")
df['clean_text']=df['clean_text'].astype(str)


vectorizer=pickle.load(open("vectorizer.pkl","rb"))
model=pickle.load(open("kmeans.pkl","rb"))


#title
st.title("Social Media Trend Analysis App")

#sidebar
st.sidebar.header("Options")
option=st.sidebar.selectbox("Choose Feature",
                            ["Search Tweets","Trending Topics","Cluster Insights"])


#search
if option =="Search Tweets":
    st.subheader("🔍Search Tweets")
    query=st.text_input("Enter keyword")

    if st.button("Search"):
        query_vec=vectorizer.transform([query])
        tfidf_matrix=vectorizer.transform(df['clean_text'])
        similarity=cosine_similarity(query_vec,tfidf_matrix)
        top_indices=similarity[0].argsort()[-5:][::-1]
        results=df.iloc[top_indices][['text','cluster']]
        st.write(results)


#trend
elif option=="Trending Topics":
    st.subheader("📈 Trends Over Time")
    df['date']=pd.to_datetime(df['date'])
    df['modified_date']=df['date'].dt.date

    trend=df.groupby(['modified_date','cluster']).size().unstack()

    st.line_chart(trend)

#cluster
elif option=="Cluster Insights":
    st.subheader("🧠 Cluster Distribution")
    cluster_counts=df['cluster'].value_counts()
    st.bar_chart(cluster_counts)