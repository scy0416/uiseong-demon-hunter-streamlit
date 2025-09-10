import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import json
import random
from models import *

if not firebase_admin._apps:
    firebase_key = st.secrets["firebase"]

    cred = credentials.Certificate(dict(firebase_key))
    firebase_admin.initialize_app(cred)

db = firestore.client()

def init_game():
    with open("map.json", "r", encoding="utf-8") as f:
        map_data = json.load(f)
    with open("monster.json", "r", encoding="utf-8") as f:
        monster_data = json.load(f)

    nodes = map_data["nodes"]
    monsters = [m["name"] for m in monster_data["monsters"]]

    for node in nodes:
        if node["id"] == 6:
            node["monster"] = "구미호"
            break

    remaining_monsters = [m for m in monster_data["monsters"]]
    available_nodes = [node for node in nodes if node["monster"] == ""]

    random.shuffle(available_nodes)

    for node, monster in zip(available_nodes, remaining_monsters):
        node["monster"] = monster

    map_str = json.dumps(map_data, ensure_ascii=False, indent=2)
    monster_str = json.dumps(monster_data, ensure_ascii=False, indent=2)

    user_data = UserData(
        health=5,
        sanity=5,
        purification=0,
        current_location="일주문",
        current_mood="",
        items="",
        choices=["다음으로"],
        story_beats=[
            StoryBeat(type="narration", speaker="", text="요괴들이 고운사를 장악했습니다!"),
            StoryBeat(type="narration", speaker="", text="선택지를 골라가며 자기만의 방식으로 요괴들을 퇴치해가보세요!")
        ],
        map=map_str,
        monster=monster_str,
        player_lore="",
        master_lore=""
    )

    doc_ref = db.collection("users").document(st.user.get("sub"))
    doc_ref.set(user_data.model_dump(), merge=True)

st.set_page_config(page_title="의성 데몬 헌터")

with st.container(border=True):
    st.title("UDH-의성 데몬 헌터")
    st.write("요괴가 창궐하는 의성, 당신의 선택이 세상을 구원합니다.")

    if not st.user.is_logged_in:
        if st.button("구글 로그인"):
            st.login()
    else:
        st.write(f"{st.user.name}님 환영합니다!")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("새 게임 시작", use_container_width=True):
                init_game()
                st.switch_page("game.py")
        with col2:
            if st.button("게임 불러오기", use_container_width=True):
                st.switch_page("game.py")

        if st.button("로그아웃"):
            st.logout()