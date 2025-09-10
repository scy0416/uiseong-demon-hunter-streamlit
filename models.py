from typing import List, TypedDict, Optional, Literal
from pydantic import BaseModel, Field

# 맵을 표현하기 위한 클래스
class World(BaseModel):
    map: str = Field(description="맵을 표현하는 json형태의 텍스트입니다.")
    monster: str = Field(description="요괴의 상태에 대한 텍스트로 json형식입니다.")

# 맵을 위한 클래스
class Map(BaseModel):
    map: str = Field(description="맵을 표현하는 json형태의 텍스트입니다.")

# 스토리 비트 클래스
class StoryBeat(BaseModel):
    type: Literal["narration", "dialogue"] = Field(description="텍스트의 타입으로 나레이션이면 narration, 대사이면 dialogue입니다.")
    speaker: str = Field(description="type이 narration이면 공백, dialogue라면 대사를 말하는 주체")
    text: str = Field(description="나레이션 또는 대사의 텍스트")

# 스토리 비트들을 위한 클래스
class StoryBeats(BaseModel):
    story_beats: List[StoryBeat] = Field(description="만들어지는 스토리 내용들")

# 사용자 데이터에 대한 클래스(상태 그래프의 상태로 사용됨)
class UserData(BaseModel):
    health: int = Field(description="플레이어의 체력입니다.")
    sanity: int = Field(description="플레이어의 정신력입니다.")
    purification: int = Field(description="고운사의 요괴 정화도 수준입니다.")
    current_location: str = Field(description="플레이어의 현재 위치입니다.")
    current_mood: str = Field(description="플레이어의 현재 기분입니다.")
    items: str = Field(description="아이템 목록으로 아이템은 컴마(,)로 구분됩니다.")
    choices: List[str] = Field(default_factory=list, description="사용자의 선택지입니다.")
    story_beats: List[StoryBeat] = Field(default_factory=list, description="게임의 진행 기록입니다.")
    map: str = Field(description="맵의 상태에 대한 텍스트로 json형식입니다.")
    monster: str = Field(description="요괴의 상태에 대한 텍스트로 json형식입니다.")
    player_lore: str = Field(description="플레이어의 기억입니다.")
    master_lore: str = Field(description="게임 전반에 걸친 기억입니다.")

# 변경하는 사용자 데이터
class ChangedUserData(BaseModel):
    health: int = Field(description="플레이어의 체력입니다.")
    sanity: int = Field(description="플레이어의 정신력입니다.")
    purification: int = Field(description="고운사의 요괴 정화도 수준입니다.")
    current_location: str = Field(description="플레이어의 현재 위치입니다.")
    current_mood: str = Field(description="플레이어의 현재 기분입니다.")
    items: str = Field(description="아이템 목록으로 아이템은 컴마(,)로 구분됩니다.")
    map: str = Field(description="맵의 상태에 대한 텍스트로 json형식입니다.")
    monster: str = Field(description="요괴의 상태에 대한 텍스트로 json형식입니다.")
    player_lore: str = Field(description="플레이어의 기억입니다.")
    master_lore: str = Field(description="게임 전반에 걸친 기억입니다.")

# 감독을 위한 클래스
class Supervising(BaseModel):
    edited_story_beat: List[StoryBeat] = Field(description="감독한 진행 내용의 최종 결과")
    supervising_again: bool = Field(description="감독이 끝나고서 추가 감독이 필요하면 True, 없으면 False")

# 선택지에 대한 클래스
class Choices(BaseModel):
    choices: List[str] = Field(description="사용자가 할 수 있는 행동들을 각각의 선택지는 성공했을 때의 이점과 실패했을 때의 벌점에 대한 내용을 포함합니다.")

# 컨텍스트
class Ctx(TypedDict):
    total_story_beats: StoryBeats
    current_story_beats: List[StoryBeat]

# 선택지 선택 모델
class SelectRequest(BaseModel):
    select: int