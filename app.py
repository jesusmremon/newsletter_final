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


st.set_page_config(page_title='Hypegenius',page_icon=':shark:', layout='wide')

col1, col2, col3 = st.columns([1, 3, 1])

serper_key = st.secrets['serper_key']
open_key = st.secrets['open_key']
browserless_key = st.secrets['browserless_key']


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

def rewritting(content, tone, educational_level):
    llm = OpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=0.7)
    template = """Based on the content provided below, you have to rewrite keep the structure intact, and only change the words. You have to rewrite it like the writer has a {tone} tone, for people with an educational level of {educational_level}
    "{content}"
    Rewriting:
    """

    prompt_template = PromptTemplate(input_variables=["tone","educational_level", "content"], template=template)

    summarizer_chain = LLMChain(llm = llm, prompt = prompt_template, verbose=False)

    summary = summarizer_chain.predict(tone = tone, educational_level = educational_level, content = content)

    return summary


def content_news(content):
    llm = OpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=0.7)
    template = """You are the best writer and journalist, the content below is a newsletter post, and based on that you have to create in a paragraph the description of the collection that this newsletter and others are part of, so if someone wanted to subscribe to all the newsletter of this collection, they will know what is about.
    "{content}"
    Description:
    """

    prompt_template = PromptTemplate(input_variables=["content"], template=template)

    summarizer_chain = LLMChain(llm = llm, prompt = prompt_template, verbose=False)

    summary = summarizer_chain.predict(content = content)

    return summary

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
        tone = st.selectbox('Tone',('Friendly', 'Professional', 'Anchor Broadcaster','1941 German Military Instructor with anger management issues', 'Lawyer'))

    submitted = st.form_submit_button("Submit")
    if submitted:
        flow_control = True
        st.toast('Generation Started')


if flow_control:

    start = time.time()

    

    with st.spinner("Generating Content"):
        result = agent({"input": query})

    
    col2.header("Newsletter Output")
    col2.info(result['output'])

    col2.write("***")

    with st.spinner("Rewriting for the tone"):
        result = rewritting(result['output'], tone, education)

    col2.header("Tone and demographic adaptation")
    col2.info(result)

    col2.write("***")
    
    with st.spinner("Rewriting for the tone"):
        result = content_news(result['output'])

    col2.header("Newsletter Description")
    col2.info(result)

    end = time.time()

    print(end - start)
    
    

