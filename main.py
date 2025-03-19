
# import os
# os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Disable Streamlit file watcher

# import warnings
# warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")  # Suppress FP16 warning

# import streamlit as st
# import whisper
# import sounddevice as sd
# import soundfile as sf
# import time
# from langchain_core.runnables import RunnableSequence
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# import librosa
# from TTS.api import TTS  # Import coqui-tts
# from pydub import AudioSegment
# from pydub.playback import play



# # Initialize coqui-tts model
# # coqui_tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False) # Example model names
# coqui_tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
# # Initialize Groq LLM
# groq_llm = ChatGroq(api_key="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", model_name="qwen-2.5-32b")

# # Load Whisper model
# @st.cache_resource
# def load_whisper_model():
#     return whisper.load_model("tiny", download_root="./models")  # Use "tiny" for faster processing

# # Function to generate interview questions
# def generate_interview_questions(job_description, resume_text):
#     prompt = PromptTemplate(
#         input_variables=["job_description", "resume_text"],
#         template="""
#         You are an experienced interviewer. Generate a list of 5 technical and behavioral interview questions based on the following job description and the candidate's resume.
#         The questions should be relevant to the role and the candidate's experience.
#         Return the questions as a numbered list without any additional text or empty lines.

#         Job Description:
#         {job_description}

#         Resume:
#         {resume_text}
#         """
#     )
#     chain = RunnableSequence(prompt | groq_llm)
#     response = chain.invoke({"job_description": job_description, "resume_text": resume_text})
    
#     # Extract the content from the AIMessage object
#     questions = response.content if hasattr(response, "content") else str(response)
    
#     # Clean the questions output
#     questions_list = [q.strip() for q in questions.split("\n") if q.strip()]
#     return questions_list

# # Function to evaluate user's response
# def evaluate_response(question, user_response):
#     prompt = PromptTemplate(
#         input_variables=["question", "user_response"],
#         template="""
#         You are an experienced interviewer. Evaluate the candidate's response to the following question and provide constructive feedback.
#         Highlight what they did well and suggest areas for improvement.

#         Question:
#         {question}

#         Candidate's Response:
#         {user_response}

#         Provide feedback in a clear and concise manner.
#         """
#     )
#     chain = RunnableSequence(prompt | groq_llm)
#     response = chain.invoke({"question": question, "user_response": user_response})
    
#     # Extract the content from the AIMessage object
#     feedback = response.content if hasattr(response, "content") else str(response)
#     return feedback

# def record_audio(filename, duration=5, sample_rate=16000):
#     st.write("Recording... Click 'Stop Recording' to finish early.")
    
#     # Check if the directory exists, if not, create it
#     audio_dir = './audio'  # You can change this to any directory path you like
#     if not os.path.exists(audio_dir):
#         os.makedirs(audio_dir)  # Creates the directory if it doesn't exist

#     file_path = os.path.join(audio_dir, filename)  # Full path to save the file
    
#     # Record audio
#     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
#     start_time = time.time()
    
#     stop_recording = False
#     stop_button = st.button("Stop Recording")
    
#     with st.empty():
#         for i in range(duration, 0, -1):
#             if stop_recording:
#                 st.write("Recording stopped.")
#                 break
#             st.write(f"Time remaining: {i} seconds")
#             time.sleep(1)
            
#             # Check if the stop button is pressed
#             if stop_button:
#                 stop_recording = True
#                 break

#     # Stop the recording if manually stopped or after the duration
#     sd.stop()

#     # Save the recorded audio to the file
#     sf.write(file_path, audio, sample_rate)  # Save audio to the specified path
#     st.write(f"Recording complete. File saved to {file_path}.")
#     return file_path  # Return the path to the recorded file
# # def play_audio(file_path):
# #     st.audio(file_path)

# # # Function to start the recording
# # def record_audio(filename, duration=120, sample_rate=16000):
# #     # Check if the directory exists, if not, create it
# #     audio_dir = './audio'  # You can change this to any directory path you like
# #     if not os.path.exists(audio_dir):
# #         os.makedirs(audio_dir)  # Creates the directory if it doesn't exist

# #     file_path = os.path.join(audio_dir, filename)  # Full path to save the file
    
# #     # Initialize session state for audio buffer
# #     if "audio_buffer" not in st.session_state:
# #         st.session_state.audio_buffer = []

# #     if "recording" not in st.session_state:
# #         st.session_state.recording = False  # Track recording state
    
# #     if "stop_button_clicked" not in st.session_state:
# #         st.session_state.stop_button_clicked = False  # Track if the stop button was clicked

# #     # Set up the button for stopping the recording
# #     stop_button = st.button("Stop Recording")
    
# #     # Handle the stop button click
# #     if stop_button:
# #         st.session_state.stop_button_clicked = True

# #     # Start the recording when the button isn't clicked yet
# #     if not st.session_state.recording and not st.session_state.stop_button_clicked:
# #         st.session_state.recording = True
# #         st.session_state.audio_buffer = []  # Reset the buffer for new recording
# #         st.write("Recording... Click 'Stop Recording' to finish early.")
    
# #     if st.session_state.recording:
# #         # Record audio in chunks
# #         audio_chunk = sd.rec(int(sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
# #         sd.wait()  # Wait for the chunk to be recorded

# #         # Append audio chunk to session state buffer
# #         st.session_state.audio_buffer.append(audio_chunk)
        
# #         # If stop button is clicked, stop recording
# #         if st.session_state.stop_button_clicked:
# #             st.session_state.recording = False
# #             st.session_state.stop_button_clicked = False  # Reset button state
# #             st.write("Recording stopped.")
# #             time.sleep(1)  # Small delay to ensure the button's state is processed before breaking
        
# #         # Display a countdown timer or status update
# #         st.write(f"Recording... Total recorded: {len(st.session_state.audio_buffer)} seconds")

# #     # Once recording is stopped, save the audio
# #     if not st.session_state.recording and len(st.session_state.audio_buffer) > 0:
# #         full_audio = np.concatenate(st.session_state.audio_buffer, axis=0)  # Concatenate all chunks

# #         # Save the recorded audio to the file
# #         try:
# #             sf.write(file_path, full_audio, sample_rate)  # Save audio to the specified path
# #             st.write(f"Recording complete. File saved to {file_path}.")
# #             play_audio(file_path)  # Play the recorded audio
# #         except Exception as e:
# #             st.write(f"Error saving the audio file: {str(e)}")

# #     return file_path  # Return the path to the recorded file

# def transcribe_audio(file_path):
#     # Load Whisper model
#     model = whisper.load_model("base")  # Using "base" model for better balance between speed and accuracy
    
#     try:
#         # Load the audio file with librosa
#         audio, _ = librosa.load(file_path, sr=16000)  # Load audio and resample to 16kHz
#         st.write(f"Audio loaded: {file_path}")
        
#         result = model.transcribe(audio)  # Transcribe the audio using Whisper
#         st.write("Transcription result obtained successfully.")
#         return result["text"]  # Return the transcribed text
#     except Exception as e:
#         st.write(f"Error during transcription: {str(e)}")
#         return None  # Return None if there is an error


# def speech_to_text():
#     try:
#         # Record and save the audio
#         audio_file = record_audio("recorded_audio.wav")  # Record audio and return file path

#         # Normalize the file path to ensure consistency across platforms
#         audio_file = os.path.abspath(audio_file)  # Get absolute path

#         # Check if the audio file exists
#         if not os.path.exists(audio_file):
#             return f"Error: The audio file {audio_file} does not exist."

#         # Transcribe audio using Whisper
#         transcription = transcribe_audio(audio_file)  # Use the transcribe_audio function
#         return transcription
#     except Exception as e:
#         return f"Error: {str(e)}"
# def play_question(question):
#     # Generate speech using coqui-tts
#     coqui_tts.tts_to_file(text=question, file_path="question.wav")
    
#     # Play audio
#     audio = AudioSegment.from_wav("question.wav")
#     play(audio)
# # Streamlit App for the Second Page
# def main():
#     # Retrieve data from the first page
#     if "resume_text" not in st.session_state or "job_description" not in st.session_state:
#         st.error("Please go back to the first page and fill out the form.")
#         return

#     resume_text = st.session_state.resume_text
#     job_description = st.session_state.job_description

#     # Generate interview questions
#     if "questions" not in st.session_state:
#         st.session_state.questions = generate_interview_questions(job_description, resume_text)
#         st.session_state.current_question_index = 0
#         st.session_state.user_response = None
#         st.session_state.feedback = None

#     # Display the current question
#     current_question = st.session_state.questions[st.session_state.current_question_index]
#     st.write(f"**Question {st.session_state.current_question_index + 1}:** {current_question}")

#     # Initialize session state for recording control
#     if "stop_recording" not in st.session_state:
#         st.session_state.stop_recording = False
#     if "recording_in_progress" not in st.session_state:
#         st.session_state.recording_in_progress = False
#     if st.button("Play Question"):
#         play_question(current_question)
#     # Start recording when "Record Response" is clicked
#     if st.button("Record Response") and not st.session_state.recording_in_progress:
#         st.session_state.recording_in_progress = True
#         st.session_state.stop_recording = False
#         st.session_state.user_response = speech_to_text()
#         st.session_state.recording_in_progress = False
#         st.write(f"**Your Response:** {st.session_state.user_response}")

#     # Stop recording button (visible only during recording)
#     # if st.session_state.recording_in_progress:
#     #     if st.button("Stop Recording"):
#     #         st.session_state.stop_recording = True
#     #         st.write("Recording stopped. Processing your response...")

#     # Evaluate the response
#     if st.session_state.user_response and st.session_state.user_response != "Error":
#         if st.button("Get Feedback"):
#             st.session_state.feedback = evaluate_response(current_question, st.session_state.user_response)
#             st.write(f"**Feedback:** {st.session_state.feedback}")

#     # Move to the next question
#     if st.button("Next Question"):
#         if st.session_state.current_question_index < len(st.session_state.questions) - 1:
#             st.session_state.current_question_index += 1
#             st.session_state.user_response = None
#             st.session_state.feedback = None
#             st.rerun()
#         else:
#             st.success("Interview completed! Great job!")

# if __name__ == "__main__":
#     main() 
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Disable Streamlit file watcher

import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")  # Suppress FP16 warning

import streamlit as st
import whisper
# import sounddevice as sd
import soundfile as sf
import time
from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import librosa
from TTS.api import TTS  # Import coqui-tts
from pydub import AudioSegment
from pydub.playback import play
import numpy as np 
import tempfile
import threading
# Initialize coqui-tts model
coqui_tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Initialize Groq LLM
groq_llm = ChatGroq(api_key="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", model_name="qwen-2.5-32b")

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny", download_root="./models")  # Use "tiny" for faster processing

# Function to generate interview questions
def generate_interview_questions(job_description, resume_text):
    prompt = PromptTemplate(
        input_variables=["job_description", "resume_text"],
        template="""
        You are an experienced interviewer. Generate a list of 5 technical and behavioral interview questions based on the following job description and the candidate's resume.
        The questions should be relevant to the role and the candidate's experience.
        Return the questions as a numbered list without any additional text or empty lines.

        Job Description:
        {job_description}

        Resume:
        {resume_text}
        """
    )
    chain = RunnableSequence(prompt | groq_llm)
    response = chain.invoke({"job_description": job_description, "resume_text": resume_text})
    
    # Extract the content from the AIMessage object
    questions = response.content if hasattr(response, "content") else str(response)
    
    # Clean the questions output
    questions_list = [q.strip() for q in questions.split("\n") if q.strip()]
    return questions_list
AUDIO_DIR = "./audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
# Function to evaluate user's response
def evaluate_response(question, user_response):
    prompt = PromptTemplate(
        input_variables=["question", "user_response"],
        template="""
        You are an experienced interviewer. Evaluate the candidate's response to the following question and provide constructive feedback.
        Highlight what they did well and suggest areas for improvement.

        Question:
        {question}

        Candidate's Response:
        {user_response}

        Provide feedback in a clear and concise manner.
        """
    )
    chain = RunnableSequence(prompt | groq_llm)
    response = chain.invoke({"question": question, "user_response": user_response})
    
    # Extract the content from the AIMessage object
    feedback = response.content if hasattr(response, "content") else str(response)
    return feedback



def transcribe_audio(file_path):
    # Load Whisper model
    model = whisper.load_model("base")  # Using "base" model for better balance between speed and accuracy
    
    try:
        # Load the audio file with librosa
        audio, _ = librosa.load(file_path, sr=16000)  # Load audio and resample to 16kHz
        st.write(f"Audio loaded: {file_path}")
        
        result = model.transcribe(audio)  # Transcribe the audio using Whisper
        st.write("Transcription result obtained successfully.")
        return result["text"]  # Return the transcribed text
    except Exception as e:
        st.write(f"Error during transcription: {str(e)}")
        return None  # Return None if there is an error


def speech_to_text():
    try:
        # Start the recording
        start_recording()
        # Stop recording once the user clicks the stop button
        stop_recording()

        # Get the file path for the saved audio
        audio_file = st.session_state['audio_file']
        
        # Check if the audio file exists
        if not os.path.exists(audio_file):
            return f"Error: The audio file {audio_file} does not exist."

        # Transcribe audio using Whisper (assuming you have a function transcribe_audio)
        transcription = transcribe_audio(audio_file)
        return transcription
    except Exception as e:
        return f"Error: {str(e)}"




def play_question(question):
    # Generate speech using coqui-tts
    coqui_tts.tts_to_file(text=question, file_path="question.wav")
    
    # Play audio
    audio = AudioSegment.from_wav("question.wav")
    play(audio)



recording = []
is_recording = False
stream = None

# def callback(indata, frames, time, status):
#     """Callback function to continuously capture audio in chunks."""
#     global recording, is_recording
#     if status:
#         print(status)
#     if is_recording:
#         recording.append(indata.copy())

# def start_recording():
#     """Starts recording audio using a callback."""
#     global recording, is_recording, stream
#     recording = []
#     is_recording = True
#     stream = sd.InputStream(samplerate=16000, channels=1, callback=callback, dtype='float32')
#     stream.start()
#     st.session_state['is_recording'] = True
#     st.write("🎙️ Recording... Click '🛑 Stop Recording' to stop.")
def callback(indata, frames, time, status):
    """Callback function to process audio data."""
    global recording, is_recording
    if status:
        print(status)
    if is_recording:
        # Instead of appending chunks from `sounddevice`, store the data to be processed
        recording.append(indata.copy())

def start_recording():
    """Starts recording audio using librosa."""
    global recording, is_recording
    recording = []
    is_recording = True
    
    # Example: Load a file and simulate chunk processing
    audio_file = 'path_to_your_audio_file.wav'  # Replace with the actual audio file path
    audio_data, sr = librosa.load(audio_file, sr=16000)  # Load audio at 16000Hz sample rate
    
    # Simulate chunk processing (use chunks similar to how `sounddevice` works)
    chunk_size = 1024
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        callback(chunk, len(chunk), None, None)  # Simulate calling the callback function

    st.session_state['is_recording'] = True
    st.write("🎙️ Processing recorded audio...")
# def stop_recording():
#     """Stops recording and saves the audio file."""
#     global is_recording, stream, recording
#     is_recording = False
#     if stream:
#         stream.stop()
#         stream.close()
    
#     if recording:
#         audio_data = np.concatenate(recording, axis=0)
#         audio_dir = "./audio"
#         os.makedirs(audio_dir, exist_ok=True)  # Ensure directory exists
#         filename = os.path.join(audio_dir, "user_response.wav")
#         sf.write(filename, audio_data, 16000)  # Save the audio file
#         st.session_state['audio_file'] = filename
#         st.success(f"✅ Recording saved: {filename}")
def stop_recording():
    global is_recording, stream, recording
    is_recording = False
    if stream:
        stream.stop()
        stream.close()
    
    if recording:
        audio_data = np.concatenate(recording, axis=0)
        
        # Save audio in a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            filename = temp_file.name
            sf.write(filename, audio_data, 16000)
            st.session_state['audio_file'] = filename
            st.success(f"✅ Recording saved: {filename}")

def main():
    st.title("🎤 AI-Powered Interview Practice")

    # Ensure session state variables exist
    if "resume_text" not in st.session_state or "job_description" not in st.session_state:
        st.error("⚠️ Please go back and fill out the form first.")
        return

    resume_text = st.session_state.resume_text
    job_description = st.session_state.job_description

    # Generate interview questions if not already present
    if "questions" not in st.session_state:
        st.session_state.questions = generate_interview_questions(job_description, resume_text)
        st.session_state.current_question_index = 0
        st.session_state.user_response = None
        st.session_state.feedback = None

    # Display current question
    current_question = st.session_state.questions[st.session_state.current_question_index]
    st.write(f"**📝 Question {st.session_state.current_question_index + 1}:** {current_question}")

    # Ensure session state tracking for recording
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "audio_file" not in st.session_state:
        st.session_state.audio_file = None

    # Play Question
    if st.button("🔊 Play Question"):
        play_question(current_question)

    # Start Recording Button
    if not st.session_state.is_recording:
        if st.button("🎙️ Start Recording"):
            start_recording()

    # Stop Recording Button - Only visible when recording
    if st.session_state.is_recording:
        if st.button("🛑 Stop Recording"):
            stop_recording()
            st.session_state.is_recording = False  # Make sure the button disappears after stop

    # Display saved audio file if available
    if st.session_state.audio_file:
        st.audio(st.session_state.audio_file, format="audio/wav")

    # Evaluate Response
    if st.session_state.audio_file:
        if st.button("📝 Get Feedback"):
            st.session_state.user_response = speech_to_text()
            if st.session_state.user_response != "Error":
                st.session_state.feedback = evaluate_response(current_question, st.session_state.user_response)
                st.write(f"**Your Response:** {st.session_state.user_response}")
                st.write(f"**Feedback:** {st.session_state.feedback}")

    # Next Question Button
    if st.button("➡️ Next Question"):
        if st.session_state.current_question_index < len(st.session_state.questions) - 1:
            st.session_state.current_question_index += 1
            st.session_state.user_response = None
            st.session_state.feedback = None
            st.session_state.audio_file = None
            st.session_state.is_recording = False  # Reset the recording state
            st.rerun()  # Rerun to clear the Stop Recording button

        else:
            st.success("✅ Interview Completed! Great Job! 🎉")


if __name__ == "__main__":
    main()