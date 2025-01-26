import streamlit as st
from PIL import Image
import io
import sys

# Function to grab image from clipboard (works on Windows/macOS)
if sys.platform == "win32" or sys.platform == "darwin":
    from PIL import ImageGrab

# Streamlit app configuration
st.title("Custom Image Resizer")
st.write("Upload an image, paste from clipboard, resize, and download the resized image.")

# Option to paste image from clipboard
if st.button('Paste from Clipboard'):
    try:
        # Grab image from clipboard
        img = ImageGrab.grabclipboard()

        if img:
            st.image(img, caption="Image from Clipboard", use_column_width=True)
        else:
            st.error("No image found in clipboard!")

    except Exception as e:
        st.error(f"Error grabbing image from clipboard: {e}")

# File uploader widget
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# If an image is uploaded or pasted from clipboard, process it
if uploaded_image is not None or ('img' in locals() and img is not None):
    # If the image was uploaded, use that, otherwise, use the clipboard image
    if uploaded_image is not None:
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
