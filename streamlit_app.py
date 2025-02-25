import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageFont
import io
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import easyocr
from sklearn.cluster import KMeans

# Check for platform and import ImageGrab accordingly
if sys.platform == "win32" or sys.platform == "darwin":
    from PIL import ImageGrab

# Streamlit app configuration
st.title("Custom Image Resizer and Editor")
st.write("Upload an image, paste from clipboard, resize, edit, and download the edited image.")

# Option to paste image from clipboard (Windows/macOS only)
if sys.platform == "win32" or sys.platform == "darwin":
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
else:
    st.write("Clipboard paste is not supported on your operating system. Please upload an image.")

# File uploader widget (for all platforms)
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

    # Image Filters
    st.sidebar.header("Image Filters")
    if st.sidebar.button("Convert to Grayscale"):
        gray_image = img.convert("L")
        st.image(gray_image, caption="Grayscale Image", use_column_width=True)

    if st.sidebar.button("Apply Sepia Filter"):
        sepia_image = np.array(img)
        sepia_image = cv2.transform(sepia_image, np.matrix([[0.272, 0.534, 0.131],
                                                            [0.349, 0.686, 0.168],
                                                            [0.393, 0.769, 0.189]]))
        sepia_image = np.clip(sepia_image, 0, 255)
        sepia_image = Image.fromarray(sepia_image.astype(np.uint8))
        st.image(sepia_image, caption="Sepia Image", use_column_width=True)

    if st.sidebar.button("Blur Image"):
        blurred_image = img.filter(ImageFilter.BLUR)
        st.image(blurred_image, caption="Blurred Image", use_column_width=True)

    if st.sidebar.button("Sharpen Image"):
        sharpened_image = img.filter(ImageFilter.SHARPEN)
        st.image(sharpened_image, caption="Sharpened Image", use_column_width=True)

    # Image Rotation
    st.sidebar.header("Image Rotation")
    angle = st.sidebar.slider("Rotate Image", min_value=0, max_value=360, value=0)
    if st.sidebar.button("Rotate"):
        rotated_image = img.rotate(angle)
        st.image(rotated_image, caption=f"Rotated Image by {angle} degrees", use_column_width=True)

    # Image Cropping
    st.sidebar.header("Image Cropping")
    left = st.sidebar.number_input("Left")
    top = st.sidebar.number_input("Top")
    right = st.sidebar.number_input("Right")
    bottom = st.sidebar.number_input("Bottom")
    if st.sidebar.button("Crop Image"):
        cropped_image = img.crop((left, top, right, bottom))
        st.image(cropped_image, caption="Cropped Image", use_column_width=True)

    # Image Flipping
    st.sidebar.header("Image Flipping")
    if st.sidebar.button("Flip Horizontally"):
        flipped_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        st.image(flipped_image, caption="Horizontally Flipped Image", use_column_width=True)

    if st.sidebar.button("Flip Vertically"):
        flipped_image = img.transpose(Image.FLIP_TOP_BOTTOM)
        st.image(flipped_image, caption="Vertically Flipped Image", use_column_width=True)

    # Change Image Format
    st.sidebar.header("Change Image Format")
    format_choice = st.sidebar.selectbox("Select Format", ["JPEG", "PNG"])
    if st.sidebar.button("Change Format"):
        formatted_image = io.BytesIO()
        img.save(formatted_image, format=format_choice)
        formatted_image.seek(0)
        st.sidebar.download_button(
            label=f"Download Image as {format_choice}",
            data=formatted_image,
            file_name=f"image.{format_choice.lower()}",
            mime=f"image/{format_choice.lower()}"
        )

    # Adjust Brightness, Contrast, Sharpness
    st.sidebar.header("Adjustments")
    brightness = st.sidebar.slider("Brightness", 0.1, 3.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0)
    sharpness = st.sidebar.slider("Sharpness", 0.1, 3.0, 1.0)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness)
    st.image(img, caption="Enhanced Image", use_column_width=True)

    # Add Text to Image
    st.sidebar.header("Add Text to Image")
    text = st.sidebar.text_input("Text to add")
    text_position_x = st.sidebar.slider("Text Position X", 0, img.width, 10)
    text_position_y = st.sidebar.slider("Text Position Y", 0, img.height, 10)
    text_color = st.sidebar.color_picker("Text Color", "#FFFFFF")
    if st.sidebar.button("Add Text"):
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((text_position_x, text_position_y), text, fill=text_color, font=font)
        st.image(img, caption="Image with Text", use_column_width=True)

    # Draw Shapes on Image
    st.sidebar.header("Draw Shapes on Image")
    shape = st.sidebar.selectbox("Select Shape", ["Rectangle", "Circle"])
    shape_color = st.sidebar.color_picker("Shape Color", "#FFFFFF")
    shape_position_x = st.sidebar.slider("Shape Position X", 0, img.width, 10)
    shape_position_y = st.sidebar.slider("Shape Position Y", 0, img.height, 10)
    shape_width = st.sidebar.slider("Shape Width", 0, img.width, 100)
    shape_height = st.sidebar.slider("Shape Height", 0, img.height, 100)
    if st.sidebar.button("Draw Shape"):
        draw = ImageDraw.Draw(img)
        if shape == "Rectangle":
            draw.rectangle([shape_position_x, shape_position_y, shape_position_x + shape_width, shape_position_y + shape_height], outline=shape_color, width=3)
        elif shape == "Circle":
            draw.ellipse([shape_position_x, shape_position_y, shape_position_x + shape_width, shape_position_y + shape_height], outline=shape_color, width=3)
        st.image(img, caption="Image with Shape", use_column_width=True)

    # Image Histogram
    st.sidebar.header("Image Histogram")
    if st.sidebar.button("Show Histogram"):
        histogram = img.histogram()
        plt.figure(figsize=(10, 4))
        plt.plot(histogram)
        st.pyplot(plt)

    # Image Metadata
    st.sidebar.header("Image Metadata")
    if st.sidebar.button("Show Metadata"):
        metadata = img.info
        st.write(metadata)

    # Apply Edge Detection
    st.sidebar.header("Edge Detection")
    if st.sidebar.button("Detect Edges"):
        edges = filters.sobel(color.rgb2gray(np.array(img)))
        st.image(edges, caption="Edge Detected Image", use_column_width=True)

    # Detect Faces in Image
    st.sidebar.header("Face Detection")
    if st.sidebar.button("Detect Faces"):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
        st.image(img_array, caption="Face Detected Image", use_column_width=True)

    # Watermark Image
    st.sidebar.header("Watermark Image")
    watermark_text = st.sidebar.text_input("Watermark Text")
    watermark_opacity = st.sidebar.slider("Watermark Opacity", 0.0, 1.0, 0.5)
    if st.sidebar.button("Add Watermark"):
        watermark = Image.new("RGBA", img.size)
        draw = ImageDraw.Draw(watermark)
        width, height = img.size
        draw.text((width - 100, height - 50), watermark_text, fill=(255, 255, 255, int(255 * watermark_opacity)))
        watermarked_image = Image.alpha_composite(img.convert("RGBA"), watermark)
        st.image(watermarked_image, caption="Watermarked Image", use_column_width=True)

    # Convert Image to ASCII Art
    st.sidebar.header("Convert to ASCII Art")
    if st.sidebar.button("Convert to ASCII"):
        reader = easyocr.Reader(['en'])
        ascii_art = reader.readtext(np.array(img), detail=0)
        st.text("\n".join(ascii_art))

    # Invert Colors
    st.sidebar.header("Invert Colors")
    if st.sidebar.button("Invert Colors"):
        inverted_image = ImageOps.invert(img.convert("RGB"))
        st.image(inverted_image, caption="Inverted Color Image", use_column_width=True)

    # Image Noise Reduction
    st.sidebar.header("Noise Reduction")
    if st.sidebar.button("Reduce Noise"):
        img_array = np.array(img)
        denoised_image = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        st.image(denoised_image, caption="Denoised Image", use_column_width=True)

    # Enhance Image Details
    st.sidebar.header("Enhance Details")
    if st.sidebar.button("Enhance Details"):
        detail_enhancer = ImageEnhance.Detail(img)
        detailed_image = detail_enhancer.enhance(2.0)
        st.image(detailed_image, caption="Detailed Image", use_column_width=True)

    # Apply Artistic Styles
    st.sidebar.header("Artistic Styles")
    style = st.sidebar.selectbox("Select Style", ["Oil Paint", "Sketch"])
    if st.sidebar.button("Apply Style"):
        if style == "Oil Paint":
            styled_image = img.filter(ImageFilter.ModeFilter(size=3))
        elif style == "Sketch":
            img_array = np.array(img.convert("L"))
            inverted_image = 255 - img_array
            blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)
            sketch_image = cv2.divide(img_array, 255 - blurred_image, scale=256)
            styled_image = Image.fromarray(sketch_image)
        st.image(styled_image, caption=f"Image with {style} Style", use_column_width=True)

    # Histogram Equalization
    st.sidebar.header("Histogram Equalization")
    if st.sidebar.button("Equalize Histogram"):
        img_array = np.array(img.convert("L"))
        equalized_image = cv2.equalizeHist(img_array)
        st.image(equalized_image, caption="Equalized Histogram Image", use_column_width=True)

    # Image Color Quantization
    st.sidebar.header("Color Quantization")
    num_colors = st.sidebar.slider("Number of Colors", 1, 256, 64)
    if st.sidebar.button("Quantize Colors"):
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_colors).fit(pixels)
        new_colors = kmeans.cluster_centers_[kmeans.labels_]
        quantized_image = new_colors.reshape(img_array.shape).astype(np.uint8)
        st.image(quantized_image, caption=f"Image with {num_colors} Colors", use_column_width=True)

    # Save Image in Different Formats
    st.sidebar.header("Save Image")
    save_format = st.sidebar.selectbox("Save As", ["JPEG", "PNG", "BMP", "GIF"])
    save_button = st.sidebar.button("Save Image")
    if save_button:
        buffered = io.BytesIO()
        img.save(buffered, format=save_format)
        buffered.seek(0)
        st.sidebar.download_button(
            label=f"Download Image as {save_format}",
            data=buffered,
            file_name=f"image.{save_format.lower()}",
            mime=f"image/{save_format.lower()}"
        )

    # Preview Image in Different Formats
    st.sidebar.header("Preview Image Formats")
    preview_format = st.sidebar.selectbox("Preview As", ["JPEG", "PNG", "BMP", "GIF"])
    if st.sidebar.button("Preview Image"):
        buffered = io.BytesIO()
        img.save(buffered, format=preview_format)
        buffered.seek(0)
        preview_image = Image.open(buffered)
        st.image(preview_image, caption=f"Image Preview as {preview_format}", use_column_width=True)

    # Display Image Size
    st.sidebar.header("Image Size")
    if st.sidebar.button("Show Size"):
        width, height = img.size
        st.write(f"Width: {width} px")
        st.write(f"Height: {height} px")

    # Image Compression
    st.sidebar.header("Image Compression")
    compression_quality = st.sidebar.slider("Compression Quality", 1, 100, 75)
    if st.sidebar.button("Compress Image"):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=compression_quality)
        compressed_image = Image.open(buffered)
        st.image(compressed_image, caption=f"Compressed Image (Quality: {compression_quality})", use_column_width=True)

    # Image Quality Adjustment
    st.sidebar.header("Image Quality Adjustment")
    quality = st.sidebar.slider("Quality", 1, 100, 75)
    if st.sidebar.button("Adjust Quality"):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        quality_adjusted_image = Image.open(buffered)
        st.image(quality_adjusted_image, caption=f"Image with Quality: {quality}", use_column_width=True)

    # Auto-Enhance Image
    st.sidebar.header("Auto-Enhance")
    if st.sidebar.button("Auto-Enhance"):
        enhanced_image = ImageOps.autocontrast(img)
        st.image(enhanced_image, caption="Auto-Enhanced Image", use_column_width=True)

    # Image Color Correction
    st.sidebar.header("Color Correction")
    if st.sidebar.button("Correct Colors"):
        color_corrected_image = ImageOps.colorize(img.convert("L"), black="black", white="white")
        st.image(color_corrected_image, caption="Color Corrected Image", use_column_width=True)

    # Image Perspective Transform
    st.sidebar.header("Perspective Transform")
    if st.sidebar.button("Transform Perspective"):
        img_array = np.array(img)
        rows, cols, ch = img_array.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        matrix = cv2.getAffineTransform(pts1, pts2)
        transformed_image = cv2.warpAffine(img_array, matrix, (cols, rows))
        st.image(transformed_image, caption="Perspective Transformed Image", use_column_width=True)

    # Warp Image
    st.sidebar.header("Warp Image")
    if st.sidebar.button("Warp Image"):
        img_array = np.array(img)
        rows, cols, ch = img_array.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250], [200, 200]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped_image = cv2.warpPerspective(img_array, matrix, (cols, rows))
        st.image(warped_image, caption="Warped Image", use_column_width=True)

    # Image Cloning
    st.sidebar.header("Clone Image")
    if st.sidebar.button("Clone Image"):
        clone_position_x = st.sidebar.slider("Clone Position X", 0, img.width, 10)
        clone_position_y = st.sidebar.slider("Clone Position Y", 0, img.height, 10)
        clone_width = st.sidebar.slider("Clone Width", 0, img.width, 100)
        clone_height = st.sidebar.slider("Clone Height", 0, img.height, 100)
        img_array = np.array(img)
        clone = img_array[clone_position_y:clone_position_y + clone_height, clone_position_x:clone_position_x + clone_width]
        img_array[clone_position_y:clone_position_y + clone_height, clone_position_x:clone_position_x + clone_width] = clone
        cloned_image = Image.fromarray(img_array)
        st.image(cloned_image, caption="Cloned Image", use_column_width=True)
