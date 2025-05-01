import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import streamlit as st
from PIL import Image

model = models.load_model('Image_classify.keras')

# Category labels
data_fruit = [
    'apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon'
]

st.title('🍎🥦 Image Classification Model')
st.write('Upload a picture of a fruit or vegetable to identify it.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated here

    img_height = 180
    img_width = 180
    image = image.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(image)
    img_batch = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_batch)
    score = tf.nn.softmax(prediction[0])

    predicted_class = data_fruit[np.argmax(score)]
    confidence = np.max(score) * 100

    st.markdown(f"### 🍎🥦 Predicted vegetable/fruit is: **{predicted_class.capitalize()}**")
    st.markdown(f"### ✅ With accuracy of: **{confidence:.2f}%**")