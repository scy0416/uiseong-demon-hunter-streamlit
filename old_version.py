from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field
import os
import json
import random

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.runtime import Runtime

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import RedirectResponse, Response
from google.oauth2 import id_token
from google.auth.transport import requests
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import credentials, firestore

#from models import *
from Gounsa.models import *

# 환경 변수 로드
load_dotenv()

# llm 모델 생성
llm = ChatOpenAI(model="gpt-4.1-2025-04-14", streaming=False)

# FastAPI 앱 생성
app = FastAPI()

# fastapi의 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 필요한 변수들 설정
GOOGLE_CLIENT_ID = os.getenv("UISEONG_GOOGLE_CLIENT_ID")
GOOGLE_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}

# 파이어베이스 초기화
cred = credentials.Certificate("uiseong2077-firebase-key.json")
firebase_admin.initialize_app(cred)

# 파이어스토어 연결
db = firestore.client()

# credential을 받아서 검증하고 사용자 정보를 반환하는 메소드
def verify_google_id_token(credential: str) -> dict:
    try:
        idinfo = id_token.verify_oauth2_token(credential, requests.Request(), GOOGLE_CLIENT_ID)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google ID token: {e}")

    if idinfo.get("iss") not in GOOGLE_ISSUERS:
        raise HTTPException(status_code=401, detail="Wrong Issuer")

    return idinfo

# 헤더로부터 로그인 사용자인지 확인하여 사용자 정보까지 반환하는 메소드
def get_current_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    return verify_google_id_token(token)

# 특정 사용자의 게임을 초기화하는 메소드
def init_game(user_id):
    # 맵 json구조 로드
    with open("Gounsa/map.json", "r", encoding="utf-8") as f:
        map_data = json.load(f)
    # 요괴 json구조 로드
    with open("Gounsa/monster.json", "r", encoding="utf-8") as f:
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

    # 사용자의 초기 데이터
    user_data = UserData(
        health=10,
        sanity=10,
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

    # 사용자의 데이터가 없으면 생성, 있으면 초기화
    doc_ref = db.collection("users").document(user_id)
    doc_ref.set(user_data.model_dump(), merge=True)
# 새로운 게임을 시작하는 엔드포인트
@app.get("/newgame")
def newgame(user: dict = Depends(get_current_user)):
    # 사용자의 id와 튜토리얼 정보를 전달해서 초기화
    init_game(user['sub'])

    # 게임 페이지로 리다이렉션
    #return RedirectResponse("http://localhost:3000/game")
    return

# 기존의 게임을 불러오는 엔드포인트
@app.get("/loadgame")
def loadgame(user: dict = Depends(get_current_user)):
    #return RedirectResponse("http://localhost:3000/game")
    return

# 게임의 정보를 불러오는 엔드포인트
@app.get("/loadinfo")
def loadinfo(user: dict = Depends(get_current_user)):
    # return {
    #     "health": 5,
    #     "sanity": 5,
    #     "purification": 50,
    #     "current_location": "일주문",
    #     "current_mood": "불안함",
    #     "items": "소금, 거울",
    #     "job": "일반인",
    #     "choices": ["[선택지1] 아무튼 선택지", "[선택지2] 아무튼 선택지", "[선택지3] 아무튼 선택지"],
    #     "history": [
    #         {"type": "narration", "text": "현재 상황이 이러이러하다."},
    #         {"type": "dialogue", "speaker": "달걀귀신", "text": "안녕"},
    #         {"type": "choice", "text": "[선택지1] 아무튼 선택지"}
    #     ]
    # }

    # 사용자 식별용 id
    user_id = user['sub']

    # 데이터 추출
    doc_ref = db.collection("users").document(user_id).get()
    # 데이터의 딕셔너리화
    user_data = doc_ref.to_dict()
    return user_data

# 선택지를 선택하는 메소드
@app.post("/selectchoice")
def selectchoice(body: SelectRequest, user: dict = Depends(get_current_user)):
    # 사용자 식별용 id
    user_id = user['sub']
    # 데이터 추출
    doc_ref = db.collection("users").document(user_id).get()
    # 사용자 데이터를 딕셔너리화
    user_data = doc_ref.to_dict()
    # UserData생성
    graph_data = UserData(**user_data)

    # 그래프 생성자
    graph_builder = StateGraph(UserData, context_schema=Ctx)

    graph_builder.add_node("스토리 진행", story_process)
    graph_builder.add_node("스토리 적용", story_apply)
    graph_builder.add_node("선택지 생성", choice_generate)
    graph_builder.add_node("서버 저장", server_apply)

    graph_builder.add_edge(START, "스토리 진행")
    graph_builder.add_edge("스토리 진행", "스토리 적용")
    graph_builder.add_edge("스토리 적용", "선택지 생성")
    graph_builder.add_edge("선택지 생성", "서버 저장")
    graph_builder.add_edge("서버 저장", END)

    graph = graph_builder.compile()

    try:
        graph.invoke(graph_data, context={"select": body.select, "user_id": user_id})
    except Exception as e:
        Response(500)

    return Response(status_code=200)

# 스토리 진행 노드
def story_process(state: UserData, runtime: Runtime[Ctx]):
    # 진행을 시키도록 만드는 프롬프트
    process_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage("""당신은 텍스트 어드벤처 게임의 진행 담당자입니다.
현재 맵에 대한 정보는 노드와 엣지로 구성된 json형식으로 제공됩니다.
이 맵에는 배치된 요괴에 대한 정보와 노드마다 사건을 기록하는 lore로 구성됩니다.
요괴들에 대한 정보는 json형식으로 제공됩니다.
요괴들의 정보에도 사건을 기록하는 lore가 있으며, 사용자가 탐색을 통해서 알아낸 퇴치법에 대한 정보도 있습니다.
사용자의 정보는 체력(health), 정신력(sanity), 현재 위치(current_location), 현재 기분(current_mood), 아이템 목록(items),
플레이어의 기억(player_lore)들이 있습니다.
세계에 대한 정보로는 진행기록(story_beats), 마스터 기억(master_lore)가 있습니다.

당신은 이 정보들을 바탕으로 게임을 진행해야 합니다.
마지막의 사용자의 선택지를 참고하여 그 선택지에 대한 적절한 진행을 해주세요.
게임 진행이 아니라 튜토리얼 상태라면 히스토리를 초기화하고 게임의 처음상태로부터 진행해주세요.
게임의 목표는 고운사의 모든 요괴를 해치우는 것으로 플레이어가 너무 헤매고 있다면
요괴를 해치울 수 있는 방향으로 이야기를 진행해주세요.

### 새로운 story_beat 생성 지침
세계에 대한 나레이션을 진행하고자 한다면 narration을,
플레이어나 요괴의 대사가 진행되는 경우에는 dialogue를 만들어주세요.
문맥이 파괴되지 않도록 조심해서 만들어주세요."""),
            SystemMessagePromptTemplate.from_template("""
### 맵
```json
{map}
```

### 요괴
```json
{monster}
```

### 사용자 정보
health: {health}
sanity: {sanity}
current_location: {current_location}
current_mood: {current_mood}
items: {items}
player_lore: {player_lore}
사용자가 선택한 행동: {player_selection}

### 세계 정보
master_lore: {master_lore}

### 진행 기록"""),
            MessagesPlaceholder("story_beats")
        ]
    )

    # 스토리 비트들을 프롬프트에 첨부 가능한 형태로 제작
    story_beats = []
    for sb in state.story_beats:
        story_beats.append({"role": "human" if sb.speaker == "player" else "assistant", "content": sb.text})

    # 체인 실행
    output = (process_prompt_template | llm.with_structured_output(StoryBeats)).invoke(
        {
            'map': state.map,
            'monster': state.monster,
            'health': state.health,
            'sanity': state.sanity,
            'current_location': state.current_location,
            'current_mood': state.current_mood,
            'items': state.items,
            'player_lore': state.player_lore,
            'master_lore': state.master_lore,
            'story_beats': story_beats,
            'player_selection': state.choices[runtime.context.get("select")]
        }
    )

    # 컨텍스트로 저장(다음 상태로 전파X)
    runtime.context['current_story_beats'] = output.story_beats

    tmp_story_beats = state.story_beats
    tmp_story_beats.append(StoryBeat(type="narration", speaker="", text=state.choices[runtime.context['select']]))
    return {"story_beats": tmp_story_beats}

# 스토리 적용 노드
def story_apply(state: UserData, runtime: Runtime[Ctx]):
    # 스토리 적용 프롬프트 템플릿
    apply_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage("""당신은 텍스트 어드벤처 게임의 문맥 적용 전문가입니다.
map/monster의 JSON을 재작성하지 마세요.
현재 맵에 대한 정보는 노드와 엣지로 구성된 json형식으로 제공됩니다.
* 맵은 nodes(id, name, monster, items, lore), edges(source, target)로 구성됩니다.
이 맵에는 배치된 요괴에 대한 정보와 노드마다 사건을 기록하는 lore로 구성됩니다.
* 요괴는 monsters로 구성되며, 각 요괴는 name, 퇴치법, lore로 구성됩니다.
요괴들에 대한 정보는 json형식으로 제공됩니다.
요괴들의 정보에도 사건을 기록하는 lore가 있으며, 사용자가 탐색을 통해서 알아낸 퇴치법에 대한 정보도 있습니다.
사용자의 정보는 체력(health), 정신력(sanity), 현재 위치(current_location), 현재 기분(current_mood), 아이템 목록(items),
플레이어의 기억(player_lore)들이 있습니다.
세계에 대한 정보로는 진행기록(story_beats), 마스터 기억(master_lore)가 있습니다.

추가적으로 현재 진행사항에 대한 정보가 제공되며, 이 기록을 바탕으로 사용자의 상태들과 맵, 요괴들의 상태들을 변경해야 합니다.
이전까지의 상태에서 현재 진행사항을 바탕으로 문맥에 맞게 상태를 변경해야 합니다.

진행사항을 적용해야하는 정보들은 다음과 같습니다.
* 플레이어의 체력(health): 변경이 없으면 그대로 전달
* 플레이어의 정신력(sanity): 변경이 없으면 그대로 전달
* 고운사의 요괴 정화도 수준(purification): 요괴를 퇴치한 경우 증가하고, 그 이외에는 변경 없이 그대로 전달하세요.
* 플레이어의 현재 위치(current_location): 현재 진행사항에서 플레이어가 이동하는 경우에 변경, 이동 이외의 행동을 하는 경우에는 그대로 전달, 사용자는 맵에 있는 노드 이외의 장소에 있을 수 없습니다.
    * 일주문은 고운사의 입구입니다.
* 플레이어의 현재 기분(current_mood): 진행사항을 참고해서 플레이어가 가질 기분을 적어주세요. 한 단어로 적어야 합니다.
* 아이템 목록(items): 현재 진행사항에서 플레이어가 특정 아이템을 얻은 것이 아니라면 변경 없이 그대로 전달
    * 각 아이템은 ','를 통해서 구분
* 맵(map): 현재 진행사항을 바탕으로 기존 맵 정보를 편집하세요.
    * 절대로 제공되는 기본 구조를 파괴해서는 안됩니다.
    * 절대로 기존의 id, name필드 데이터는 변경해서는 안됩니다.
    * 새로운 노드의 추가, 기존 노드의 삭제는 안됩니다.
    * 각 노드의 순서를 변경하고, 연결 관계를 재정의해서는 안됩니다.
    * 변경사항이 적용되는 맵은 이전 맵과 진행상황을 고려해서 변경되어야 하며, 이 때, 문맥이 맞아야 합니다.
    * 노드에 있던 요괴를 갑자기 없애거나 만들어내지 마세요.
    * 변경사항이 없다면 그대로 전달하세요.
* 요괴(monster): 현재 진행사항에서 요괴에 대한 변경할 사항이 있다면 편집하고, 이외에는 그대로 전달하세요.
    * [주의] 절대로 제공되는 monster의 기본구조를 파괴해서는 안됩니다.
* 플레이어의 기억(player_lore): 플레이어가 가억할 내용이 있다면 그 내용을 추가하고, 없다면 그대로 전달하세요.
    * 무조건 제공되는 내용에서 추가되어야 하며, 갑자기 큰 변화가 있어서는 안됩니다.
* 게임 전반에 걸친 기억(master_lore): 게임 전반에서 기억해야 할 내용이 있다면 그 내용을 추가하고, 없다면 그대로 전달하세요.
    * 무조건 제공되는 내용에서 추가되어야 하며, 갑자기 큰 변화가 있어서느 안됩니다.""")
            , SystemMessagePromptTemplate.from_template("""### 맵
```json
{map}
```

### 요괴
```json
{monster}
```

### 사용자 정보
health: {health}
sanity: {sanity}
current_location: {current_location}
current_mood: {current_mood}
items: {items}
player_lore: {player_lore}

### 세계 정보
master_lore: {master_lore}

### 지금까지의 진행사항"""),
            MessagesPlaceholder("story_beats"),
            SystemMessage("""### 현재 진행사항(적용 대상)"""),
            MessagesPlaceholder("current_story_beats")
        ]
    )

    # 삽입할 지금까지의 스토리 비트와 현재 스토리비트 정제
    story_beats = []
    for sb in state.story_beats:
        story_beats.append({"role": "human" if sb.speaker == "player" else "assistant", "content": sb.text})
    current_story_beats = []
    for sb in runtime.context['current_story_beats']:
        current_story_beats.append({"role": "human" if sb.speaker == "player" else "assistant", "content": sb.text})

    # 체인 실행
    output = (apply_prompt_template | llm.with_structured_output(ChangedUserData)).invoke(
        {
            'map': state.map,
            'monster': state.monster,
            'health': state.health,
            'sanity': state.sanity,
            'current_location': state.current_location,
            'current_mood': state.current_mood,
            'items': state.items,
            'player_lore': state.player_lore,
            'master_lore': state.master_lore,
            'story_beats': story_beats,
            'current_story_beats': current_story_beats
        }
    )

    # 현재 진행 스토리 체인 적용
    output_dict = output.model_dump()
    #output_dict['story_beats'] = state.story_beats
    output_dict['story_beats'] = state.story_beats + runtime.context['current_story_beats']

    # 상태 전파
    return output_dict

# 선택지 생성 노드
def choice_generate(state: UserData, runtime: Runtime[Ctx]):
    # 선택지 생성 프롬프트 템플릿
    choices_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage("""당신은 텍스트 어드벤처 게임의 선택지 생성 전부가입니다.
현재 맵에 대한 정보는 노드와 엣지로 구성된 json형식으로 제공됩니다.
* 맵은 nodes(id, name, monster, items, lore), edges(source, target)로 구성됩니다.
* edges는 source, target으로 구성됩니다.
이 맵에는 배치된 요괴에 대한 정보와 노드마다 사건을 기록하는 lore로 구성됩니다.
* 요괴는 monsters로 구성되며, 각 요괴는 name, 퇴치법, lore로 구성됩니다.
요괴들에 대한 정보는 json형식으로 제공됩니다.
요괴들의 정보에도 사건을 기록하는 lore가 있으며, 사용자가 탐색을 통해서 알아낸 퇴치법에 대한 정보도 있습니다.
사용자의 정보는 체력(health), 정신력(sanity), 현재 위치(current_location), 현재 기분(current_mood), 아이템 목록(items),
플레이어의 기억(player_lore)들이 있습니다.
세계에 대한 정보로는 진행기록(story_beats), 마스터 기억(master_lore)가 있습니다.

추가적으로 현재까지의 진행사항에 대한 정보가 제공되며, 이 기록들을 바틍으로 사용자가 할 수 있는 행위 선택지를 2~5개까지 만들어야 합니다.

이동하는 선택지는 항상 하나 이상을 만들어주세요.
요괴들을 해치울 수 있는 방향의 선택지들을 만들어주세요.

제공되는 정보들은 다음과 같습니다.
* 플레이어의 체력(health)
* 플레이어의 정신력(sanity)
* 고운사의 요괴 정화도 수준(purification)
* 플레이어의 현재 위치(current_location)
* 플레이어의 현재 기분(current_mood)
* 아이템 목록(items)
* 맵(map)
* 요괴(monster)
* 플레이어의 기억(player_lore)
* 게임 전반에 걸친 기억(master_lore)"""),
            SystemMessagePromptTemplate.from_template("""### 맵
```json
{map}
```

### 요괴
```json
{monster}
```

### 사용자 정보
health: {health}
sanity: {sanity}
current_location: {current_location}
current_mood: {current_mood}
items: {items}
player_lore: {player_lore}

### 세계 정보
master_lore: {master_lore}

### 지금까지의 진행사항"""),
            MessagesPlaceholder("story_beats")
        ]
    )

    # 스토리 진행사항 정제
    story_beats = []
    for sb in state.story_beats:
        story_beats.append({"role": "human" if sb.speaker == "player" else "assistant", "content": sb.text})

    # 체인 실행
    output = (choices_prompt_template | llm.with_structured_output(Choices)).invoke(
        {
            'map': state.map,
            'monster': state.monster,
            'health': state.health,
            'sanity': state.sanity,
            'current_location': state.current_location,
            'current_mood': state.current_mood,
            'items': state.items,
            'player_lore': state.player_lore,
            'master_lore': state.master_lore,
            'story_beats': story_beats
        }
    )

    # 상태 전파
    return output

# 서버 적용 노드
def server_apply(state: UserData, runtime: Runtime[Ctx]):
    # 컨텍스트에서 user_id추출
    user_id = runtime.context.get("user_id")

    # 사용자 정보 업데이트
    doc_ref = db.collection("users").document(user_id)
    doc_ref.update(state.model_dump())