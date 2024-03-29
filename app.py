import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import os

## Function To get response from LLAma 2 model

def getLLamaresponse(input_text,no_words,blog_style):

    ### LLama2 model
    model_path = r'C:\Users\Anju Reddy K\Personal_projects\Blog_Generation\models\llama-2-7b-chat.ggmlv3.q8_0.bin'
    print(f"Model Path: {model_path}")
    llm = CTransformers(model=model_path,model_type='llama',config={'max_new_tokens': 256, 'temperature': 0.01})

    
    ## Prompt Template

    template="""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response






st.set_page_config(page_title="AI Personal Assistant",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("AI Personal Assistant 🤖")

input_text=st.text_input("Enter the Prompt Below")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('Number of Words')
with col2:
    blog_style=st.selectbox('Select Phrase',
                            ('Technical','Business','Personal'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))
    
st.markdown("[Link to Code!!](https://github.com/Anjureddyk/Personal_Assistant)", unsafe_allow_html=True)
