import streamlit as st
import os
import openai

st.set_page_config(page_title="GPT-3 Demo")

st.title("GPT-3 Demo")

st.text("Demo GPT-3 with Streamlit")

# Load your API key
openai.api_key = os.environ.get("OPENAI_API_KEY")


def main():

    selected_box = st.sidebar.selectbox(
        "Choose one of the following",
        ("Instruct", "Chat", "Q&A", "Essay Writer", "TL;DR Summarization", "Essay Outline")
    )

    if selected_box == "Instruct":
        st.header("Instruct GPT-3")
        max_tokens = 150

        try:
            form = st.form(key="my_form5")
            command = form.text_area(label="Enter instructions here", value="Write a poem",height=200)
            max_tokens = st.number_input("Response Length", max_value=3048, value=max_tokens, step=1)
            submit_button = form.form_submit_button(label="Submit")

            if submit_button:
                st.header("Result")
                answer = instruct_gpt3(command, max_tokens)
                command += answer
                st.write(answer)

            
        except Exception as e:
            st.success(f'Something went wrong! {e}')
            
    if selected_box == "Chat":
        st.header("Chat with GPT-3")
        max_tokens = 150

        try:
            form = st.form(key="my_form2")
            command = form.text_area(label="Enter some conversation here", value="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:",height=200)
            max_tokens = st.number_input("Response Length", max_value=2048, value=max_tokens, step=1)
            submit_button = form.form_submit_button(label="Submit")

            if submit_button:
                st.header("Result")
                answer = chat_gpt3(command, max_tokens)
                command += answer
                st.write(answer)

            
        except Exception as e:
            st.success(f'Something went wrong! {e}')
    
    if selected_box == "Q&A":
        st.header("Q&A with GPT-3")
        try:
            form = st.form(key="my_form4")
            command = form.text_area(label="Enter some conversation here", value="I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer.\nQ: \nA:",height=200)
            submit_button = form.form_submit_button(label="Submit")

            if submit_button:
                st.header("Result")
                answer = qna_gpt3(command)

                st.write(answer)

        except Exception as e:
            st.success(f'Something went wrong! {e}')

    if selected_box == "Classification":
        st.header("Classification with GPT-3")
        try:
            form = st.form(key="my_form")
            command = form.text_area(label="Give the title of your essay", value="Uses of GPT in education. \n", height=200)
            submit_button = form.form_submit_button(label="Submit")

            if submit_button:
                st.header("Result")
                answer = classification_gpt3(command)

                st.write(answer)
        except Exception as e:
            st.success(f'Something went wrong! {e}')       

    if selected_box == "TL;DR Summarization":
        st.header("TL;DR with GPT-3")
        max_tokens = 60

        try:
            form = st.form(key="my_form")
            command = form.text_area(label="Enter some text here", value="**insert text in here and don't delete tl;dr below**\n\n tl;dr", height=200)
            max_tokens = st.number_input("Response Length", max_value=2048, value=max_tokens, step=1)
            submit_button = form.form_submit_button(label="Submit")

            if submit_button:
                st.header("Result")
                answer = tldr_gpt3(command)

                st.write(answer)
        except Exception as e:
            st.success(f'Something went wrong! {e}')

    if selected_box == "Essay Outline":
        st.header("Essay with GPT-3")
        max_tokens = 60

        try:
            form = st.form(key="my_form2")
            command = form.text_area(label="Enter some command here", height=200)
            max_tokens = st.number_input("Response Length", max_value=2048, value=max_tokens, step=1)
            submit_button = form.form_submit_button(label="Submit")

            if submit_button:
                st.header("Result")
                answer = command + essay_gpt3(command, max_tokens)

                st.write(answer)
        
        except Exception as e:
            st.success(f'Something went wrong! {e}')

def instruct_gpt3(prompt, max_tokens):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7, 
        max_tokens=2050,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    answer = response.choices[0]['text']
    return answer
            
def chat_gpt3(prompt, max_tokens):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.9, 
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " AI:"]
    )

    answer = response.choices[0]['text']
    return answer

def qna_gpt3(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    answer = response.choices[0]['text']
    return answer

def classification_gpt3(prompt):
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=3400,
        top_p=1,
        frequency_penalty=0.3,
        presence_penalty=0.3,
        stop=["\n"]
    )
    answer = response.choices[0]['text']
    return answer

def tldr_gpt3(prompt):
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    answer = response.choices[0]['text']
    return answer

def essay_gpt3(prompt, max_tokens):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    answer = response.choices[0]['text']
    return answer

if __name__ == "__main__":
    main()
