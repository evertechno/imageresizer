import streamlit as st
from PIL import Image

# Streamlit app configuration
st.title("Custom Image Resizer")
st.write("Upload an image and specify the new size to resize the image.")

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

        # Save the resized image
        save_button = st.sidebar.button("Save Resized Image")
        if save_button:
            # Save to a local file
            resized_image_path = "resized_image.jpg"
            resized_image.save(resized_image_path)
            st.sidebar.write(f"Image saved at: {resized_image_path}")
