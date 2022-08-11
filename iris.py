# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 01:39:16 2022

@author: TOURE Hachirou
"""
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import streamlit.components.v1 as ct
import time

st.sidebar.header('Entrer les informations de votre iris')
ct.html("<h1 style='color:green;text-align:center'>Application de prédiction du type de l'iris</h1>") 
def entrer():
    sepal_len=st.sidebar.slider('Sepal Hauteur ',4.2,7.9,5.4)
    sepal_width=st.sidebar.slider('Sepal Largeur',2.0,4.4,4.0)
    petal_len=st.sidebar.slider('Petal Longueur',1.0,6.9,1.2)
    petal_width=st.sidebar.slider('Petal Largeur',0.1,2.5,0.2)
    data={'sepal_long':sepal_len,
          'sepal_larg':sepal_width,
          'petal_long':petal_len,
          'petal_larg':petal_width
          
        }
    fet=pd.DataFrame(data,index=[0])
    return fet
df=entrer()
st.subheader('Vos données')
st.dataframe(df)
@ st.cache
def data():
    iris=datasets.load_iris()
    return iris
def fiting():
    iris=data() 
    X=iris.data
    Y=iris.target
    mod=RandomForestClassifier()
    mod=mod.fit(X,Y)
    return mod  
prediction=fiting().predict(df)
predi_proba=fiting().predict_proba(df)
 
with st.spinner('Patientez...'):
    st.subheader('Prédiction')
    time.sleep(1)
    st.write(data().target_names[prediction])
       
    st.subheader('Probabilité de Prédiction')
    col1,col2=st.columns(2) 
    col1.write(data().target_names) 
    col2.write(predi_proba)    