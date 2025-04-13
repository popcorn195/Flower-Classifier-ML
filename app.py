#streamlit run app.py
#run and save
import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

st.header('Flower Classification CNN Model')
flower_names=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model=load_model('/content/Flowers_Recog_Model.h5')

def classify_images(image_path):
  input_image=tf.keras.utils.load_img(image_path,target_size=(180,180)) #Load and resize
  input_image_array=tf.keras.utils.img_to_array(input_image) #Convert to array
  
  input_image_exp_dim=tf.expand_dims(input_image_array,0) #Add batch dim

  predictions=model.predict(input_image_exp_dim) #Model inference
  result=tf.nn.softmax(predictions[0]) #Convert logits to probabilities

  predicted_class = flower_names[np.argmax(result)]
  score = round(100 * np.max(result), 2)

  outcome='The image belong to '+predicted_class+'with a score of '+str(score)
  return outcome

  uploaded_file=st.file_uploader('Upload an Image')
  if uploaded_file is not None:
    with open(os.path.join('upload',uploaded_file.name),'wb') as f:
      f.write(uploaded_file.getbuffer())

    st.image(uploaded_file,width=200)

  st.markdown(classify_images(uploaded_file))
