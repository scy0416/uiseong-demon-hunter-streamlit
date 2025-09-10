import streamlit as st
from langchain_openai import ChatOpenAI
import firebase_admin
from firebase_admin import credentials, firestore
import time

llm = ChatOpenAI(model="gpt-4.1-2025-04-14", streaming=True, api_key=st.secrets["openai"]["api_key"])

if not st.user.is_logged_in:
    st.switch_page("title.py")

sub = st.user.get("sub")
name = st.user.get("name")

if not firebase_admin._apps:
    creds = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(creds)

db = firestore.client()

def load_data():
    doc_ref = db.collection("users").document(sub).get()
    user_data = doc_ref.to_dict()
    #print(user_data)

    st.session_state['health'].write(f"체력: {user_data['health']}")
    st.session_state['sanity'].write(f"정신력: {user_data['sanity']}")
    st.session_state['purification'].write(f"정화도: {user_data['purification']}")
    st.session_state['current_mood'].write(f"현재 기분: {user_data['current_mood']}")
    st.session_state['current_location'].write(f"현재 위치: {user_data['current_location']}")

    for story_beat in user_data['story_beats']:
        #print(story_beat)
        with st.session_state.story_beats:
            if story_beat['type'] == 'narration':
                with st.chat_message("ai"):
                    st.write(story_beat['text'])
            else:
                with st.chat_message("human"):
                    st.write(story_beat['text'])
    def generate_button(label):
        if st.button(label, use_container_width=True):
            process(label)

    with st.session_state.buttons:
        with st.container():
            #st.button("버튼1", use_container_width=True)
            #st.button("버튼2", use_container_width=True)
            #st.button("버튼3", use_container_width=True)
            generate_button("기이인 텍스트 버튼1")
            generate_button("버튼2")
            generate_button("버튼3")

def str_generator(string):
    for c in string:
        time.sleep(0.05)
        yield c

def process(selection):
    with st.session_state.story_beats:
        with st.chat_message("human"):
            st.write_stream(str_generator(selection))

st.set_page_config("의성 데몬 헌터", layout="wide")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    with st.container(border=True):
        st.session_state.story_beats = st.container()
        st.session_state.buttons = st.empty()

with col2:
    with st.container(border=True):
        st.title("현재 상태")
        st.session_state.health = st.empty()
        st.session_state.sanity = st.empty()
        st.session_state.purification = st.empty()
        st.session_state.current_mood = st.empty()
        st.session_state.current_location = st.empty()

load_data()