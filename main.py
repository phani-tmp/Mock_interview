
# import os
# os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Disable Streamlit file watcher

# import warnings
# warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")  # Suppress FP16 warning
# import asyncio
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
# import numpy as np 
# import tempfile
# import threading
# # Initialize coqui-tts model
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
# AUDIO_DIR = "./audio"
# os.makedirs(AUDIO_DIR, exist_ok=True)
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
#         # Start the recording
#         start_recording()
#         # Stop recording once the user clicks the stop button
#         stop_recording()

#         # Get the file path for the saved audio
#         audio_file = st.session_state['audio_file']
        
#         # Check if the audio file exists
#         if not os.path.exists(audio_file):
#             return f"Error: The audio file {audio_file} does not exist."

#         # Transcribe audio using Whisper (assuming you have a function transcribe_audio)
#         transcription = transcribe_audio(audio_file)
#         return transcription
#     except Exception as e:
#         return f"Error: {str(e)}"




# def play_question(question):
#     # Generate speech using coqui-tts
#     coqui_tts.tts_to_file(text=question, file_path="question.wav")
    
#     # Play audio
#     audio = AudioSegment.from_wav("question.wav")
#     play(audio)



# recording = []
# is_recording = False
# stream = None
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



# def main():
#     st.title("🎤 AI-Powered Interview Practice")

#     # Ensure session state variables exist
#     if "resume_text" not in st.session_state or "job_description" not in st.session_state:
#         st.error("⚠️ Please go back and fill out the form first.")
#         return

#     resume_text = st.session_state.resume_text
#     job_description = st.session_state.job_description

#     # Generate interview questions if not already present
#     if "questions" not in st.session_state:
#         st.session_state.questions = generate_interview_questions(job_description, resume_text)
#         st.session_state.current_question_index = 0
#         st.session_state.user_response = None
#         st.session_state.feedback = None

#     # Display current question
#     current_question = st.session_state.questions[st.session_state.current_question_index]
#     st.write(f"**📝 Question {st.session_state.current_question_index + 1}:** {current_question}")

#     # Ensure session state tracking for recording
#     if "is_recording" not in st.session_state:
#         st.session_state.is_recording = False
#     if "audio_file" not in st.session_state:
#         st.session_state.audio_file = None

#     # Play Question
#     if st.button("🔊 Play Question"):
#         play_question(current_question)

#     # Start Recording Button
#     if not st.session_state.is_recording:
#         if st.button("🎙️ Start Recording"):
#             start_recording()

#     # Stop Recording Button - Only visible when recording
#     if st.session_state.is_recording:
#         if st.button("🛑 Stop Recording"):
#             stop_recording()
#             st.session_state.is_recording = False  # Make sure the button disappears after stop

#     # Display saved audio file if available
#     if st.session_state.audio_file:
#         st.audio(st.session_state.audio_file, format="audio/wav")

#     # Evaluate Response
#     if st.session_state.audio_file:
#         if st.button("📝 Get Feedback"):
#             st.session_state.user_response = speech_to_text()
#             if st.session_state.user_response != "Error":
#                 st.session_state.feedback = evaluate_response(current_question, st.session_state.user_response)
#                 st.write(f"**Your Response:** {st.session_state.user_response}")
#                 st.write(f"**Feedback:** {st.session_state.feedback}")

#     # Next Question Button
#     if st.button("➡️ Next Question"):
#         if st.session_state.current_question_index < len(st.session_state.questions) - 1:
#             st.session_state.current_question_index += 1
#             st.session_state.user_response = None
#             st.session_state.feedback = None
#             st.session_state.audio_file = None
#             st.session_state.is_recording = False  # Reset the recording state
#             st.rerun()  # Rerun to clear the Stop Recording button

#         else:
#             st.success("✅ Interview Completed! Great Job! 🎉")


# if __name__ == "__main__":
#     asyncio.run(main())
# import os
# import streamlit as st
# from streamlit_mic_recorder import mic_recorder
# import whisper
# from langchain_core.runnables import RunnableSequence
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# import librosa
# from TTS.api import TTS  # Import coqui-tts
# from pydub import AudioSegment
# from pydub.playback import play
# import numpy as np 
# import speech_recognition as sr
# import tempfile
# import soundfile as sf 
# import wave
# from io import BytesIO

# # Disable Streamlit file watcher
# os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# # Suppress FP16 warning
# import warnings
# warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# # Initialize coqui-tts model
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

# # Ensure the './audio' directory exists
# AUDIO_DIR = "./audio"
# os.makedirs(AUDIO_DIR, exist_ok=True)

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

# def play_question(question):
#     # Generate speech using coqui-tts
#     coqui_tts.tts_to_file(text=question, file_path="question.wav")
    
#     # Play audio
#     audio = AudioSegment.from_wav("question.wav")
#     play(audio)

# def main():
#     st.title("🎤 AI-Powered Interview Practice")

#     # Ensure session state variables exist
#     if "resume_text" not in st.session_state or "job_description" not in st.session_state:
#         st.error("⚠️ Please go back and fill out the form first.")
#         return

#     resume_text = st.session_state.resume_text
#     job_description = st.session_state.job_description

#     # Generate interview questions if not already present
#     if "questions" not in st.session_state:
#         st.session_state.questions = generate_interview_questions(job_description, resume_text)
#         st.session_state.current_question_index = 0
#         st.session_state.user_response = None
#         st.session_state.feedback = None
#         st.session_state.audio_file = None
#         st.session_state.is_playing = False  # Add session state for playing question

#     # Display current question
#     current_question = st.session_state.questions[st.session_state.current_question_index]
#     st.write(f"**📝 Question {st.session_state.current_question_index + 1}:** {current_question}")

#     # Play Question
#     if st.button("🔊 Play Question"):
#         st.session_state.is_playing = True
#         play_question(current_question)
#         st.session_state.is_playing = False  # Reset after playing

#     # Start and stop recording using st.audio_input()
#     audio_file = st.audio_input("Record your response")

#     if audio_file and not st.session_state.is_playing:  # Ensure recording doesn't happen while playing
#         # Save the uploaded audio file to disk
#         filename = os.path.join(AUDIO_DIR, f"user_response_{st.session_state.current_question_index}.wav")
#         with open(filename, "wb") as f:
#             f.write(audio_file.getvalue())
        
#         recognizer = sr.Recognizer()
#         transcription = "It is not clear, can you please explain what is happening?"
        
#         # Load the audio file using SpeechRecognition
#         with sr.AudioFile(filename) as source:
#             audio_data = recognizer.record(source)

#         try:
#             # Use Google Web Speech API for transcription
#             transcription = recognizer.recognize_google(audio_data)
#             st.write(f"Transcription: {transcription}")

#         except Exception as e:
#             st.write(f"Error during transcription: {e}")
        
#         # Transcribe the recorded audio
#         st.session_state.user_response = transcription
#         if st.session_state.user_response != "Error":
#             st.write(f"**Your Response:** {st.session_state.user_response}")

#     # Evaluate Response
#     if st.session_state.user_response and st.session_state.user_response != "Error":
#         if st.button("📝 Get Feedback"):
#             st.session_state.feedback = evaluate_response(current_question, st.session_state.user_response)
#             st.write(f"**Feedback:** {st.session_state.feedback}")

#     # Next Question Button
#     if st.button("➡️ Next Question"):
#         if st.session_state.current_question_index < len(st.session_state.questions) - 1:
#             st.session_state.current_question_index += 1
#             st.session_state.user_response = None
#             st.session_state.feedback = None
#             st.session_state.audio_file = None
#             st.rerun()  # Rerun to clear the previous recording and outputs
#         else:
#             st.success("✅ Interview Completed! Great Job! 🎉")

# if __name__ == "__main__":
#     main() 
import os
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from TTS.api import TTS  # Import coqui-tts
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import speech_recognition as sr
import tempfile
import soundfile as sf
import wave
from io import BytesIO

# Disable Streamlit file watcher
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# Suppress FP16 warning
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

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

def play_question(question):
    # Generate speech using coqui-tts
    coqui_tts.tts_to_file(text=question, file_path="question.wav")
    
    # Play audio
    audio = AudioSegment.from_wav("question.wav")
    play(audio)

# def main():
#     st.title("🎤 AI-Powered Interview Practice")

#     # Ensure session state variables exist
#     if "resume_text" not in st.session_state or "job_description" not in st.session_state:
#         st.error("⚠️ Please go back and fill out the form first.")
#         return

#     resume_text = st.session_state.resume_text
#     job_description = st.session_state.job_description

#     # Generate interview questions if not already present
#     if "questions" not in st.session_state:
#         st.session_state.questions = generate_interview_questions(job_description, resume_text)
#         st.session_state.current_question_index = 0
#         st.session_state.user_response = None
#         st.session_state.feedback = None
#         st.session_state.audio_file = None
#         st.session_state.is_playing = False  # Add session state for playing question

#     # Display current question
#     current_question = st.session_state.questions[st.session_state.current_question_index]
#     st.write(f"**📝 Question {st.session_state.current_question_index + 1}:** {current_question}")

#     # Play Question
#     if st.button("🔊 Play Question"):
#         st.session_state.is_playing = True
#         play_question(current_question)
#         st.session_state.is_playing = False  # Reset after playing

#     # Start and stop recording using st.audio_input()
#     audio_file = st.audio_input("Record your response")

#     if audio_file and not st.session_state.is_playing:  # Ensure recording doesn't happen while playing
#         # Use a temporary file for storing the recorded audio
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             temp_file.write(audio_file.getvalue())
#             temp_filename = temp_file.name
#         st.session_state.audio_file=audio_file
#         recognizer = sr.Recognizer()
#         transcription = "It is not clear, can you please explain what is happening?"

#         # Load the audio file using SpeechRecognition
#         with sr.AudioFile(temp_filename) as source:
#             audio_data = recognizer.record(source)

#         try:
#             # Use Google Web Speech API for transcription
#             transcription = recognizer.recognize_google(audio_data)
#             st.write(f"Transcription: {transcription}")

#         except Exception as e:
#             st.write(f"Error during transcription: {e}")

#         # Transcribe the recorded audio and store it in session state
#         st.session_state.user_response = transcription
#         if st.session_state.user_response != "Error":
#             st.write(f"**Your Response:** {st.session_state.user_response}")

#         # Evaluate Response
#         if st.session_state.audio_file:
#             if st.button("📝 Get Feedback"):
                
#                 if st.session_state.user_response != "Error":
#                     st.session_state.feedback = evaluate_response(current_question, st.session_state.user_response)
#                     st.write(f"**Your Response:** {st.session_state.user_response}")
#                     st.write(f"**Feedback:** {st.session_state.feedback}")

#         # Next Question Button
#         if st.button("➡️ Next Question"):
#             if st.session_state.current_question_index < len(st.session_state.questions) - 1:
#                 st.session_state.current_question_index += 1
#                 st.session_state.user_response = None
#                 st.session_state.feedback = None
#                 st.session_state.audio_file = None
#                 st.session_state.is_recording = False  # Reset the recording state
#                 st.rerun()  # Rerun to clear the Stop Recording button
#             else:
#                 st.success("✅ Interview Completed! Great Job! 🎉")
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
        st.session_state.audio_file = None
        st.session_state.is_playing = False  # Add session state for playing question

    # Display current question
    current_question = st.session_state.questions[st.session_state.current_question_index]
    st.write(f"**📝 Question {st.session_state.current_question_index + 1}:** {current_question}")

    # Play Question
    if st.button("🔊 Play Question"):
        st.session_state.is_playing = True
        play_question(current_question)
        st.session_state.is_playing = False  # Reset after playing

    # Start and stop recording using st.audio_input()
    audio_file = st.audio_input("Record your response")

    if audio_file and not st.session_state.is_playing:  # Ensure recording doesn't happen while playing
        # Use a temporary file for storing the recorded audio
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio_file.getvalue())
            temp_filename = temp_file.name
        st.session_state.audio_file = audio_file
        recognizer = sr.Recognizer()
        transcription = "It is not clear, can you please explain what is happening?"

        # Load the audio file using SpeechRecognition
        with sr.AudioFile(temp_filename) as source:
            audio_data = recognizer.record(source)

        try:
            # Use Google Web Speech API for transcription
            transcription = recognizer.recognize_google(audio_data)
            st.write(f"Transcription: {transcription}")

        except Exception as e:
            st.write(f"Error during transcription: {e}")

        # Transcribe the recorded audio and store it in session state
        st.session_state.user_response = transcription
        if st.session_state.user_response != "Error":
            st.write(f"**Your Response:** {st.session_state.user_response}")

        # Evaluate Response
        if st.session_state.audio_file:
            if st.button("📝 Get Feedback"):
                
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
