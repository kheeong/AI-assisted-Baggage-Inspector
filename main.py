import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras import preprocessing
import numpy as np
from enum import Enum
from io import BytesIO, StringIO
from typing import Union
import streamlit as st
import cv2
import pandas as pd
from PIL import Image
image_1 = Image.open('danger1.png')
image_2 = Image.open('nodanger1.png')
image_3 = Image.open('danger2.png')
image_4 = Image.open('nodanger2.jpg')
STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""
dsize = (224, 224)
FILE_TYPES = ["png", "jpg"]

model = tf.keras.models.load_model('model_2.h5')

def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader("Upload file", type=FILE_TYPES)
    show_file = st.empty()   
    show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES) + " or choose a picture from below")
    col1, col2,col3,col4 = st.beta_columns(4)
    col1.image(image_1, use_column_width=True)
    col2.image(image_2, use_column_width=True)
    col3.image(image_3, use_column_width=True)
    col4.image(image_4, use_column_width=True)
    b1,b2,b3,b4 = st.beta_columns(4)
    if b1.button('Select picture 1'):
        resized = preprocessing.image.load_img("danger1.png", target_size=(224, 224)) 
    elif b2.button('Select picture 2'):
        resized = preprocessing.image.load_img("nodanger1.png", target_size=(224, 224)) 
    elif b3.button('Select picture 3'):
        resized = preprocessing.image.load_img("danger2.png", target_size=(224, 224)) 
    elif b4.button('Select picture 4'):
        resized = preprocessing.image.load_img("nodanger2.jpg", target_size=(224, 224)) 
    else:
        if not file:
            return
        data = file.read()
        show_file.image(data)
        nparr = np.fromstring(data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        resized = cv2.resize(img_np,dsize)
        file.close()
    input_arr = preprocessing.image.img_to_array(resized)
    img=preprocess_input(input_arr)
    img_arr = np.array([img])
    predictions = model.predict(img_arr)
    print(predictions)
    result = np.where(predictions[0] == np.amax(predictions[0]))
    print(result[0][0])
    if(result[0][0] == 0):
        print("Dangerous item found")
        st.markdown("<h1 style='text-align: center; color: red;'>Dangerous item found</h1>", unsafe_allow_html=True)
    else:
        print("No dangerous item found")
        st.markdown("<h1 style='text-align: center; color: green;'>No dangerous item found</h1>", unsafe_allow_html=True)
    st.write()
    df = pd.DataFrame({"Dangerous item chance":[predictions[0][0],"{0:.0%}".format(predictions[0][0])] , "No dangerous item chance":[predictions[0][1],"{0:.0%}".format(predictions[0][1])]})
    st.dataframe(df.assign(hack='').set_index('hack'))

main()