import streamlit as st
from PIL import Image
import io

# Streamlit app configuration
st.title("Custom Image Resizer")
st.write("Upload an image, specify the new size, and download the resized image.")

# File uploader widget
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# If an image is uploaded, process it
if uploaded_image is not None:
    # Open image using PIL
    img = Image.open(uploaded_image)
    
    # Display original image
    st.image(img, caption="Original Image", use_column_width=True)

    # User input for resizing
    st.sidebar.header("Resize Settings")
    width = st.sidebar.number_input("New Width (px)", min_value=1, value=img.width)
    height = st.sidebar.number_input("New Height (px)", min_value=1, value=img.height)

    # Resize the image based on user input
    if st.sidebar.button("Resize Image"):
        resized_image = img.resize((width, height))
        st.image(resized_image, caption=f"Resized Image: {width}x{height}", use_column_width=True)

        # Prepare the resized image for download
        # Save to a BytesIO object
        buffered = io.BytesIO()
        resized_image.save(buffered, format="PNG")
        buffered.seek(0)

        # Provide a download link for the resized image
        st.sidebar.download_button(
            label="Download Resized Image",
            data=buffered,
            file_name="resized_image.png",
            mime="image/png"
        )
