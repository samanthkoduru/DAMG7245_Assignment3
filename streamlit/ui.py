from numpy import array
import streamlit as st 
import requests



# interact with FastAPI endpoint
backend = "http://127.0.0.1:8000/docs#/default/create_upload_file_users_me_uploadfile_post"

import pickle
reg_model = pickle.load(open("models.sav", 'rb'))



with st.sidebar: 
    st.title("This App enables users to upload images of 4 different modalities of a storm event and predict number of flashes")
    choice = st.radio("Navigation", ["Predict #flashes"])

if choice == "Predict #flashes":
    st.title("Please upload images of 4 different modalities of a storm event")
    ir069 = st.file_uploader("Upload ir069 image")
    ir107 = st.file_uploader("Upload ir107 image")
    vis = st.file_uploader("Upload vis image")
    vil = st.file_uploader("Upload vil image")
    if st.button("Nuumber of Flashes",key="#flashes"):
            
        col1, col2 = st.columns(2)
        files=[('files',(vis.name,vis,'image/png')),('files',(ir069.name,ir069,'image/png')),('files',(vil.name,vil,'image/png')),('files',(ir107.name,ir107,'image/png'))]
        segments = requests.post(backend, files=files)
        flashes=segments.json()
        print(segments.json())
        col1.header("Flashes")
        col1.write(flashes['flashes'])