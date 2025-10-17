# Courtesy of https://github.com/Hazl-Gallery/A-Face-Swap-App
# In your venv
# $ streamlit run app.py

import streamlit as st
import requests
import base64
from PIL import Image
import io
import os
import numpy as np
import cv2
#import mediapipe as mp
#mp_selfie_segmentation = mp.solutions.selfie_segmentation
from test import get_swap_ab_image_as_cvimg
from helper import detect_face_in_cvframe, detect_face_in_pilimage
from io import BytesIO

st.set_page_config(page_title="Batch Face Swap App", layout="wide")
st.title("Batch Face Swap Application")

# Create two columns for image upload
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Swap Image")
    swap_image = st.file_uploader("Choose the face to swap from", type=['jpg', 'jpeg', 'png'])
    if swap_image:
        st.image(swap_image, caption="Swap Image", use_column_width=True)

with col2:
    st.subheader("Upload Inputs")
    input_images = st.file_uploader("Choose the targets", type=['jpg', 'jpeg', 'png', 'avi', 'mp4', 'mkv'], accept_multiple_files=True)
    if input_images:
        st.write(f"Selected {len(input_images)} targets for processing")
        # Display a sample of uploaded images (up to 3)
        cols = st.columns(min(3, len(input_images)))
        for idx, col in enumerate(cols):
            if idx < len(input_images):
                with col:
                    # Determine if input_images[idx] is image or video, call st.component correspondingly
                    if input_images[idx].type in ['image/jpg', 'image/jpeg', 'image/png']:
                        st.image(input_images[idx], caption=f"Input Image {idx + 1}", use_column_width=True)
                    elif input_images[idx].type in ['video/avi', 'video/mp4', 'video/mkv']:
                        st.video(input_images[idx])


def encode_image_to_base64(image_file):
    if image_file is not None:
        if isinstance(image_file, Image.Image):
            im_file = BytesIO()
            image_file.save(im_file, format="JPEG")
            image_file = im_file

        # Read the file into bytes
        bytes_data = image_file.getvalue()
        # Encode to base64
        base64_str = base64.b64encode(bytes_data).decode('utf-8')
        # Get file extension, defaulting to jpg if not available
        file_ext = "jpg"
        if hasattr(image_file, 'type') and image_file.type:
            file_ext = image_file.type.split('/')[-1]
        return f"data:image/{file_ext};base64,{base64_str}"
    return None

def face_swap_local(swap_image_data, input_image_data):

    def _formsubmit_to_img(image_data, format="PIL"):
        # Remove the header if it exists
        if "," in image_data:
            image_data = image_data.split(",")[1]
 
        # Decode the base64 string
        simg_bytes = base64.b64decode(image_data)
        
        # # Convert to a NumPy array
        # nparr = np.frombuffer(simg_bytes, np.uint8)
        # cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to a PIL image
        byte_stream = io.BytesIO(simg_bytes)

        # Open the image using Pillow
        pil_image = Image.open(byte_stream)

        return pil_image

    simg = _formsubmit_to_img(swap_image_data)
    iimg = _formsubmit_to_img(input_image_data)    
    
    output_cvimg = get_swap_ab_image_as_cvimg(simg, iimg)
    return output_cvimg


# Create a button to perform batch face swap
if st.button("Perform Batch Face Swap"):
    if swap_image is None:
        st.error("Please upload the swap image (the source of swap)!")

    else:
        if not input_images and not input_videos:
            st.error("Please upload at least one input (the target of swap)!")
        else:
            # Encode swap image once
            swap_image_data = encode_image_to_base64(swap_image)
            
            # Create a progress bar for img
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create a container for results
            results_container = st.container()
            with results_container:
                st.subheader("Results")
                # Calculate number of columns needed (max 3 per row)
                num_cols = min(3, len(input_images))
                results_cols = st.columns(num_cols)
                
                # Process each image
                for idx, input_image in enumerate(input_images):

                    # Update progress
                    progress = (idx + 1) / len(input_images)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing input {idx + 1} of {len(input_images)}...")
                    
                    # Determine if input_image is image or video
                    #print(input_image.type)

                    if input_image.type in ['image/jpeg','image/jpg','image/png']:
                        # Detect faces in input image
                        #print(type(input_image))
                        inputpil = Image.open(input_image)
                        #print(type(inputpil))
                        face_img, xywh = detect_face_in_pilimage(pilimage=inputpil)
                        Wf, Hf = face_img.size
                        face_img = face_img.resize((128, 128))

                        # Encode input image
                        input_image_data = encode_image_to_base64(face_img) # face_img as PIL
                        
                        # Make the API request
                        result = face_swap_local(swap_image_data, input_image_data)
                        if result is not None: 
                            try:
                                result = cv2.resize(result, (Wf, Hf))
                                #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)       
                                # Convert result to pil from cv

                                result_image = Image.fromarray(result)
                                #result_image.save("result_img.png")
                                # Paste result back to originial pil
                                inputpil.paste(result_image, (xywh[0], xywh[1]))   
                                #inputpil.save("pasted_img.png")                       
                                # Display the result in the appropriate column
                                col_idx = idx % num_cols
                                with results_cols[col_idx]:
                                    st.image(inputpil, caption=f"Result {idx + 1}", use_column_width=True)
                            except Exception as e:
                                col_idx = idx % num_cols
                                with results_cols[col_idx]:
                                    st.error(f"Failed to process result for target {idx + 1}: {str(e)}")
                        else:
                            col_idx = idx % num_cols
                            with results_cols[col_idx]:
                                st.error(f"Failed to process target {idx + 1}")

                    elif input_image.type in ['video/avi','video/mp4','video/mkv']:
                        # Process video
                        temp_video_path = "temp_uploaded_video.mp4" 
                        with open(temp_video_path, "wb") as f:
                            f.write(input_image.read())
                        st.success(f"Video saved temporarily as: {temp_video_path}")

                        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                            vidcap = cv2.VideoCapture(temp_video_path)
                            frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = int(vidcap.get(cv2.CAP_PROP_FPS))

                            # 2. Initialize a VideoWriter object
                            output_video_path = 'temp_output_video.mp4'
                            fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # You can choose other codecs like 'MP4V'
                            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

                            frame_number = 0
                            success, frame = vidcap.read()

                            while success:

                                # Detect and crop face in frame
                                face_img, xywh = detect_face_in_cvframe(cvframe=frame) # face_img as PIL

                                # Process the current frame (e.g., display it, apply filters)
                                st.write(f"Processing frame: {frame_number}")
                                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for PIL
                                
                                if face_img:
                                    # Encode input image
                                    input_image_data = encode_image_to_base64(face_img)
                            
                                    # Make the API request
                                    result = face_swap_local(swap_image_data, input_image_data) # result is cv_img
                                    #print(result.shape)
                                    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                                    frame[xywh[1]:xywh[1]+xywh[3], xywh[0]:xywh[0]+xywh[2]] = cv2.resize(result, (xywh[2], xywh[3]))
                                    
                                    # Write replaced frame to new video
                                    out.write(frame)

                                # Read the next frame
                                success, frame = vidcap.read()
                                frame_number += 1

                            vidcap.release() # Release the video capture object
                            out.release()
                            os.remove(temp_video_path) # Clean up the temporary file
                            st.success("Video processing complete and temporary input file removed.")
                            st.video(output_video_path)

                    else:
                        st.error("Unsupported datatype for input, only jpg/png/avi/mp4/mkv supported")
                
                # Complete the progress bar
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")

            