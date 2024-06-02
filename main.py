import streamlit as st
from keras.models import load_model
from PIL import Image

from utilities.util import classify, set_background , image_to_base64






set_background('./bgs/bg1.jpeg')

# set title
st.markdown("<h1 style='text-align: center; color: white;'>Colon Cancer Detection</h1>", unsafe_allow_html=True)

# set header
st.markdown("<h2 style='text-align: center; color: white;'>Please upload an image</h2>", unsafe_allow_html=True)

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/best_model.h5')

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')


    # Set image size
    image_size = (300, 300)  # Adjust the size as needed

    # Add padding and border radius

    st.markdown(
        f'<div style="display: flex; justify-content: center;">'
        f'<div style="padding: 10px; border-radius: 10px; background-color: white;">'
        f'<img src="data:image/jpeg;base64,{image_to_base64(image)}" '
        f'style="width: {image_size[0]}px; height: auto; border-radius: 10px;">'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # classify image
    class_name, conf_score = classify(image, model, class_names)


    # Write classification with centered text and white color
    if class_name == "colon_aca":
        st.markdown(
            f'<div style="text-align: center; color: white;">'
            f'<h2 style="color: white;"> The image is predicted to have colon cancer.</h2>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="text-align: center; color: white;">'
            f'<h2 style="color: white;"> The image is predicted not to have colon cancer.</h2>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div style="text-align: center; color: white;">'
        f'<h2 style="color: white;">class_name : {class_name}</h2>'
        f'<h3 style="color: white;">score: {int(conf_score * 1000) / 10}%</h3>'
        f'</div>',
        unsafe_allow_html=True
    )

