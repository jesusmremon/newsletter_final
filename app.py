import openai
import requests
import json
from typing import Type
from bs4 import BeautifulSoup
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain.schema import SystemMessage
from pydantic import BaseModel, Field
from langchain import OpenAI, LLMChain, PromptTemplate
import os

import streamlit as st
from langchain.callbacks import get_openai_callback
import time
import datetime
from datetime import date


st.set_page_config(page_title='Hypegenius',page_icon=':shark:', layout='wide')

col1, col2, col3 = st.columns([1, 3, 1])

serper_key = st.secrets['serper_key']
open_key = st.secrets['open_key']
browserless_key = st.secrets['browserless_key']

os.environ['OPEN_AI_KEY'] = open_key



    
## Functions Definition
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "tbs": "qdr:w"
    })

    headers = {
        'X-API-KEY': serper_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    # print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={browserless_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        # print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", open_api_key = open_key)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="Your objective is to get the highest quality data to create a newsletter")
    url: str = Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, or data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message_blog = SystemMessage(
    content = """You are a Newsletter Witter, The user will give you some information, about the topic, you should use that and change the user input where it corresponds below, with that, you will write a newsletter using the structure you have at the bottom. Your output should be 700 words long and in markdown format.\n\nThe structure must be the next one, replace my <> with the content it says\n\n<Title>\n\n<Introduction title>\n<Introduction content>\n\n<Main Body title>\n<Main Body content>\n\n<Conclusion Title>\n<Conclusion content>

    Here are the conditions and rules you must follow:
    1/ The Newsletter must be engaging, informative with good data
    2/ The Newsletter has to be around 800 words long
    3/ The Newsletter must address the topic really well
    5/ The Newsletter needs to be written in a way that is easy to read and understand
    6/ The Newsletter needs to give the audience insights
    7/ Be as precise as possible giving useful information
    8/ Make the content easy to read and entertaining
    9/ Do not make things up, use only the information you have
    Newsletter:
    """
    )

system_message_news = SystemMessage(
    content = """ You are a world-class journalist, able to inform about any news and create engaging articles worth a Pulitzer prize. You will write 5 different articles with the information you have. Write the output in markdown format to make it easier to read

    Here are the conditions and rules you must follow:
    1/ You have to write 5 articles
    2/ The articles must be different, do not repeat the same article twice
    3/ The articles must be engaging and easy to read
    5/ Each article should have a catchy title and content with 200 words
    6/ The articles need to give the audience insights
    7/ Be as precise as possible giving useful information
    8/ Make the content easy to read and entertaining
    9/ Do not make things up, use only the information you have
    10/ Add the URLs that you have used to provide more context to the reader
    Newsletter:
    """
    )

system_message_novel = SystemMessage(
    content = """ You are a world-class writter, able to write and create books based in any content. You will write a novel with the information you have. Write the output in markdown format to make it easier to read

    Here are the conditions and rules you must follow:
    1/ You have to write a novel based on the topic
    2/ The novel must be genuine and entertaining to read
    3/ The novel must be engaging and easy to read
    4/ The novel should have a catchy title and content with 700 words, you can create as many chapters as you want
    5/ The novel needs to give the audience insights
    6/ Be as precise as possible giving useful information
    7/ Make the content easy to read and entertaining
    8/ Do not make things up, use only the information you have
    Newsletter:
    """
    )


system_message_guide = SystemMessage(
    content = """ You are a world-class writter, able to write and create guides based in any content. You will write a guide with the information you have. Write the output in markdown format to make it easier to read

    Here are the conditions and rules you must follow:
    1/ You have to write a guide based on the topic
    2/ The guide must be genuine and entertaining to read
    3/ The guide must be engaging and easy to read
    4/ The guide should have a catchy title, content with 700 words explaining everything and giving all the context, and a bullet point to summarize.
    5/ The guide needs to give the audience insights
    6/ Be as precise as possible giving useful information
    7/ Make the content easy to read and entertaining
    8/ Do not make things up, use only the information you have
    Newsletter:
    """
    )


system_message_comic = SystemMessage(
    content = """ You are a world-class comic writter, able to write and create comics based in any content. You will write a comic with the information you have. Write the output in markdown format to make it easier to read

    Here are the conditions and rules you must follow:
    1/ You have to write the comic dialogs and a sentence describing each scene so we can draw it
    2/ The comic must be genuine and entertaining to read
    3/ The comic must be engaging and easy to read
    4/ The comic should have at least 10 different scenes and various character
    5/ Create for each scene and character a prompt with enough information to draw it
    6/ Be as precise as possible giving useful information
    7/ Make the content easy to read and entertaining
    8/ Do not make things up, use only the information you have
    9/ Describe the scenes as much as possible so a drawer can paint them as well as the characters aesthetic
    Newsletter:
    """
    )




def rewritting(content, tone, educational_level, open_key):
    template = "Based on the content provided by the user, you have to rewrite keep the structure intact, and only change the words."
    
    completion = client.chat.completions.create(
      model="gpt-4-1106-preview",
      messages=[
        {"role": "system", "content": template},
        {"role": "user", "content": "You have to rewrite the content provided below like the writer has a {tone} tone, for people with an educational level of {educational_level}: CONTENT: {content}".format(tone = tone, educational_level=educational_level, content=content)}
      ]
    )

    rewritte = completion.choices[0].message.content

    return rewritte


def content_news(content, open_key):
    llm = OpenAI(model_name="gpt-4-1106-preview", temperature=0.7, openai_api_key=st.secrets['open_key'])
    template = """You are the best writer and journalist, the content below is a newsletter post, and based on that you have to create in a paragraph the description of the collection that this newsletter and others are part of, so if someone wanted to subscribe to all the newsletter of this collection, they will know what is about, be as generic as possible without missing any key information.
    "{content}"
    Description:
    """

    prompt_template = PromptTemplate(input_variables=["content"], template=template)

    summarizer_chain = LLMChain(llm = llm, prompt = prompt_template, verbose=False)

    summary = summarizer_chain.predict(content = content)

    return summary


flow_control = False

hide_menu = """
    <style>
    .st-emotion-cache-zq5wmm {visibility: hidden;}
    #MainMenu {visibility: hidden;}
        footer{
            visibility: hidden;
        }

        footer:after {
            content: 'Made by the one and only Jesus Remon';
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
        }
    </style>            
"""

st.markdown(hide_menu, unsafe_allow_html=True)

# st.markdown(""" <style> .font {
#             font-size:40px ; 
#             font-family: 'IBM Plex Sans', sans-serif; 
#             color: #262626; 
#             text-align: center;
#             } 
#             </style> """, unsafe_allow_html=True)

# st.sidebar.markdown('<p class="font">Hypegenius</p>', unsafe_allow_html=True)


col2.title('Hypegenius')
col2.write('###')

education = 'High School'
tone = 'Friendly'

query = "How to do mindfulness"

with st.sidebar:
    st.header("LLM parameters")
    advanced = st.toggle('Advance Parameters Selection')


with col2.form('query'):
    query = st.text_input('Introduce the topic to search:', value = query)
    
    if advanced:
        education = st.selectbox('Target educational level',('Middle School', 'High School', 'College','Phd'))
        tone = st.selectbox('Tone',('Friendly', 'Professional', 'Anchor Broadcaster','Serius', 'Lawyer'))
        type = st.selectbox('Type of Newsletters',('Blog Style', 'News Style', 'Guide Style', 'Novel Style', 'Comic Style (Waiting for Dalle 3 to finish)'))

    submitted = st.form_submit_button("Submit")
    if submitted:
        flow_control = True
        st.toast('Generation Started')


if flow_control:

    if type == 'News Style':
        system_message = system_message_news
    elif type == 'Guide Style':
        system_message = system_message_guide
    elif type == 'Novel Style':
        system_message = system_message_novel
    elif type == 'Comic Style (Waiting for Dalle 3 to finish)':
        system_message = system_message_comic
    elif type == 'Blog Style':
        system_message = system_message_blog

    start = time.time()

    agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
    }
    
    llm = ChatOpenAI(temperature=0.7, model="gpt-4-1106-preview", openai_api_key=open_key)
    memory = ConversationSummaryBufferMemory(
        memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)
    
    agent = initialize_agent(
        tools,
        llm,
        agent = AgentType.OPENAI_FUNCTIONS,
        verbose = False,
        agent_kwargs=agent_kwargs,
        memory=memory
    )

    
    with st.spinner("I'm generating the Content"):
        result = agent({"input": query})

    col2.info(result['output'])

    col2.write("***")

    with st.spinner("Let me rewrite for the tone"):
        result = rewritting(result['output'], tone, education, open_key)

    col2.header("Tone and demographic adaptation")
    col2.info(result)

    col2.write("***")
    
    with st.spinner("I'm generating the concept"):
        result = content_news(result, open_key)

    col2.header("Newsletter Description")
    col2.info(result)

    end = time.time()

    print(end - start)

        
    

