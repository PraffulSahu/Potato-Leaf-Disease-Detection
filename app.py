import streamlit as st
import time
import tensorflow as tf
import numpy as np
import gdown
import os
file_id='1leolmjb0jvwGscMbrVJ_vImY4aHqrsM4'
url='https://drive.google.com/uc?id=1leolmjb0jvwGscMbrVJ_vImY4aHqrsM4'
model_path='trained_plant_disease_model.keras'

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive....")
    gdown.download(url, model_path, quiet=False)

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebarstreamlit runstr
st.sidebar.title("🌿 Plant Disease Detection System for Sustainable Agriculture")
st.sidebar.write("To start, change the **<span style='background-color:#FFD700; color:black; padding:3px; border-radius:5px;'> Page below ⬇</span>**", unsafe_allow_html=True)


app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])
#app_mode = st.sidebar.selectbox("Select Page",["Home"," ","Disease Recognition"])

# import Image from pillow to open images
from PIL import Image
img = Image.open("image.jpeg")

# display image using streamlit
# width is used to set the width of an image
st.image(img)

#Main Page
if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>🌱 Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)
    st.write("""Welcome to the Potato Leaf Disease Prediction System, an AI-powered platform designed to help farmers and agricultural experts detect and diagnose potato plant diseases with ease. Using advanced machine learning and image processing techniques, our system analyzes leaf images to identify diseases early, ensuring timely intervention and improved crop health.

### Key Features:
✅ **AI-Powered Disease Detection** – Upload leaf images and receive instant disease predictions.  
✅ **User-Friendly Interface** – Simple and intuitive design for farmers and researchers.  
✅ **Comprehensive Disease Database** – Learn about common potato diseases, symptoms, and treatment options.  
✅ **Real-Time Insights** – Get recommendations for disease management and prevention.  
✅ **Accessible Anywhere** – Use our web-based platform on mobile and desktop devices.  
""")
    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("🔬 Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("⚠️ Please Upload a Leaf Image for Analysis:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    
    #Predict button
    if(st.button("🔍 Predict")):
        #st.snow()
        with st.spinner("⏳ Processing... Please Wait!"):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)  # Simulate loading
                progress_bar.progress(percent_complete + 1)
            st.success("✅ Process Completed!")
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
