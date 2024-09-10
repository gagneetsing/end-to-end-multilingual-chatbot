# import streamlit as st
# import openai
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.llms import Ollama
# import os

# import os
# from dotenv import load_dotenv
# load_dotenv()

# ## Langsmith Tracking
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With Ollama"

# ## Prompt Template
# prompt=ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are an AI that provides answers in fluent and grammatically correct Hindi. When given a question in English, first interpret the question, then respond accurately in Hindi.")
# ,
#         ("user","Question:{question}")
#     ]
# )

# def generate_response(question,llm,temperature,max_tokens):
#     llm=Ollama(model=llm)
#     output_parser=StrOutputParser()
#     chain=prompt|llm|output_parser
#     answer=chain.invoke({'question':question})
#     return answer

# ## #Title of the app
# st.title("Enhanced Q&A Chatbot With OpenAI")


# ## Select the OpenAI model
# llm=st.sidebar.selectbox("Select Open Source model",["gemma2"])

# ## Adjust response parameter
# temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
# max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# ## MAin interface for user input
# st.write("Goe ahead and ask any question")
# user_input=st.text_input("You:")


# if user_input :
#     response=generate_response(user_input,llm,temperature,max_tokens)
#     st.write(response)
# else:
#     st.write("Please provide the user input")


import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from gtts import gTTS
import pygame
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Prompt Template to ensure Hindi responses
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Regardless of the user's input language, always respond in punjabi.",
        ),
        ("user", "Question: {question}"),
    ]
)


# Function to play speech
def speak_text(text, language="pa"):
    # Generate speech from the text using gTTS
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("response.mp3")

    # Play the speech using pygame
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()


# Function to generate the chatbot's response (in Hindi)
def generate_response(question, llm, temperature, max_tokens):
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer


# Streamlit interface for the chatbot
st.title("Speaking Q&A Chatbot With punjabi Responses")

# Sidebar parameters for adjusting model settings
llm = st.sidebar.selectbox("Select Open Source model", ["gemma2"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    # Generate the response based on user input
    response = generate_response(user_input, llm, temperature, max_tokens)

    # Display the text response in Hindi
    st.write(response)

    # Convert the text response to speech and play it
    speak_text(response)
else:
    st.write("Please provide the user input.")
