import streamlit as st
from PIL import Image
import numpy as np


if st.button('Images'):
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    if image is not None:
        st.image(
            image,
            caption=f"Uploaded image {img_array.shape[0:2]}",
            use_column_width=True,
        )

    img = st.file_uploader("")