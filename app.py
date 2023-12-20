import streamlit as st
import pyttsx3
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

def getLLamaresponse(input_text, no_words, blog_style):
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                       model_type='llama',
                       config={'max_new_tokens': 256,
                               'temperature': 0.01})

    template = """
        Create content for a {blog_style} job profile on the topic of {input_text}
        within {no_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)

    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

st.set_page_config(page_title="Personal Assistant",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Personal Assistant 🤖")

input_text = st.text_input("How can I assist you today?")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Maximum Words')
with col2:
    blog_style = st.selectbox('Context',
                              ('Professional', 'Casual', 'Technical'), index=0)

submit = st.button("Generate")

if submit:
    response = getLLamaresponse(input_text, no_words, blog_style)
    st.write(response)

    # Text-to-speech
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()
