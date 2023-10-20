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

import streamlit as st
from langchain.callbacks import get_openai_callback
import time
import datetime
from datetime import date


st.set_page_config(page_title='Hypegenius',page_icon=':brain:')


## Functions Definition
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
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

    print("Scraping website...")
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
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

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
        description="The objective & task that users give to the agent")
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
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content = """You are a twitter influencer, you are the best at creating informative and engaging tweets about any specific topic, you know how to make any tweet to go viral, be friendly and professional when writting.
    I want you to write a twitter thread about the topic.

    Here are the conditions and rules you must follow:
    1/ The thread must be engaging, informative with good data
    2/ The thread needs to be around 3-5 tweets
    4/ The thread needs to be viral, and get 1000 likes
    5/ The thread need to be written in a way that is easy to read and undertand
    6/ The thread needs to give audience actionable advice and insights too
    7/ Be frienly and engagind, it has to be easy to read
    8/ DO NOT use emojis
    9/ Do not use hashtags
    """
    )

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo-16k-0613", openai_api_key=open_key)
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

serper_key = st.secrets['serper_key']
open_key = st.secrets['open_key']
browserless_key = st.secrets['browserless_key']

openai.api_key = open_key

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


st.title('Hypegenius')
st.write('###')

age_group = (15, 70)
location = 'us'
gender = 'Both'
tone = 'Friendly'

query = "How to do mindfulness"

with st.sidebar:
    st.header("LLM parameters")
    advanced = st.toggle('Advance Parameters Selection')


with st.form('query'):
    query = st.text_input('Introduce the topic to search:', value = query)
    
    if advanced:
        age_group = st.slider('Select the age range', 15, 100, (18, 25))
        location = st.selectbox('Audience Country',('🇨🇦', '🇪🇸', '🇺🇸'))
        education = st.selectbox('Target educational level',('Middle School', 'High School', 'College','Phd'))
        tone = st.selectbox('Tone',('Friendly', 'Professional', 'Anchor Broadcaster','1941 German Military Instructor with anger management issues', 'Lawyer'))

    submitted = st.form_submit_button("Submit")
    if submitted:
        flow_control = True
        st.toast('Generation Started')


if flow_control:

    if location == '🇨🇦':
        location = 'Canada'
        search_location = 'ca'
        
    elif location == '🇪🇸':
        location = 'Spain'
        search_location = 'es'

    elif location == '🇺🇸':
        location = 'United States'
        search_location = 'us'

   

    start = time.time()
    
    result = agent({"input": query})

    st.info(result['output'])

    end = time.time()

    print(end - start)
    
    

