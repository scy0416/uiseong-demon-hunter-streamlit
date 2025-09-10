import streamlit as st

st.set_page_config(page_title="의성 데몬 헌터")

st.title("UDH-의성 데몬 헌터")
st.write("의성의 신비로운 요괴들을 물리치고 고대 사찰들을 탐험하는 액션 어드벤쳐 게임")

st.divider()

with st.container(border=True):
    st.header("게임 개요")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("게임 특징")
        st.write("의성 지역의 실제 사찰들을 배경으로 한 몰입감 있는 게임플레이")
        st.write("한국 전통 요괴들과의 전투 시스템")
        st.write("다양한 무기와 스킬로 요괴들을 물리치는 액션")

    with col2:
        st.subheader("스토리")
        st.write("의성 지역에 나타난 요괴들을 물리치고 고대 사찰들의 평화를 지키는 데몬헌터가 되어 신비로운 모험을 떠나세요. 각 사찰마다 숨겨진 비밀과 강력한 요괴들이 기다리고 있습니다.")

with st.container(border=True):
    st.subheader("게임 트레일러")
    st.video("https://youtu.be/F6VVrFvDHO8?si=qjOL8rTs-w2qLzpx")

if st.button("게임 플레이하러 가기", use_container_width=True):
    st.switch_page("title.py")