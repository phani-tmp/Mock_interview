import os
import streamlit as st
import requests
import base64
import time
from TTS.api import TTS

# Disable Streamlit file watcher to avoid Torch inspection issues
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# D-ID API Key (Replace this with your actual API key)
DID_API_KEY = "cGhhbmluZHJhMTk3QGdtYWlsLmNvbQ:wwwSEvx6yOADZ-RA3sDaj"

# Function to generate speech using Coqui TTS
def generate_speech(question_text, output_audio="question_audio.wav"):
    try:
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        tts.tts_to_file(text=question_text, file_path=output_audio, use_phonemes=False)
        return output_audio
    except Exception as e:
        st.error(f"❌ Error generating speech: {e}")
        return None

# Function to convert file to Base64
def encode_file_to_base64(file_path):
    try:
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"❌ Error encoding file to Base64: {e}")
        return None

# Function to generate lip-sync video using D-ID API
def generate_lip_sync_video(image_base64, audio_base64):
    url = "https://api.d-id.com/talks"
    
    headers = {
        "Authorization": f"Bearer {DID_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "source_url": f"data:image/jpeg;base64,{image_base64}",
        "audio_url": f"data:audio/wav;base64,{audio_base64}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        response_data = response.json()

        if "id" in response_data:
            video_id = response_data["id"]

            # Wait for video processing
            time.sleep(5)

            # Fetch the video URL
            video_url = f"https://api.d-id.com/talks/{video_id}"
            return video_url
        else:
            st.error(f"❌ Failed to generate video: {response_data.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"❌ Error generating lip-sync video: {e}")
        return None

# Streamlit UI
st.title("🤖 AI Interviewer")
st.write("Upload an image, enter a question, and see the interviewer respond!")

uploaded_image = st.file_uploader("Upload Interviewer Image", type=["jpg", "png"])
question = st.text_area("Enter Interview Question")

if st.button("Generate Interview"):
    if uploaded_image and question:
        st.write("🎙️ Generating Speech...")
        
        # Convert image to Base64
        image_base64 = base64.b64encode(uploaded_image.read()).decode("utf-8")

        # Generate Speech
        audio_path = generate_speech(question)
        if audio_path:
            audio_base64 = encode_file_to_base64(audio_path)

            if audio_base64:
                st.write("🎥 Generating Lip-Sync Video...")
                
                # Generate Lip Sync Video
                video_url = generate_lip_sync_video(image_base64, audio_base64)

                if video_url:
                    st.video(video_url)
                else:
                    st.error("❌ Failed to generate lip sync video. Please try again.")
            else:
                st.error("❌ Failed to encode audio file.")
        else:
            st.error("❌ Failed to generate speech.")
    else:
        st.error("❌ Please upload an image and enter a question.")