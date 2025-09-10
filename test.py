import streamlit as st

import firebase_admin
from firebase_admin import credentials, firestore

@st.cache_resource
def get_db():
    creds = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(creds)
    return firestore.client()

db = get_db()

#print(db.collection("user"))

if not st.user.is_logged_in:
    if st.button("로그인"):
        st.login()
else:
    if st.button("로그아웃"):
        st.logout()