import streamlit as st

st.set_page_config(page_title="의성 데몬 헌터")

pages = {
    "메뉴": [
        st.Page("title.py", title="플레이 페이지"),
        st.Page("landing.py", title="랜딩 페이지"),
        st.Page("game.py", title="게임 페이지")
    ]
}

pg = st.navigation(pages, position="top")
pg.run()