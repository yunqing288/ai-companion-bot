"""记忆相关的 prompt（A1：结构化提取）。

红线：语义理解与结构化字段提取全部由模型完成，本地不做关键词推断。
本文件只存放 prompt 文本与拼装函数，不含任何业务逻辑。

事件 schema：每条事件 = 通用三件套（category/content/date）+ 因类别而异的 fields。
- content：完整细节（细节层 + 向量召回用）。
- fields：结构化字段，状态层只渲染其中一部分（见 core/state_layer.py）。
"""

# 各类别需要模型额外提取的结构化字段（写进 fields）：
#   relationship_milestone 关系转折 : who_initiated 谁主动 / from_relation 旧关系 / to_relation 新关系
#   her_life              用户信息  : sub_type=基础个人信息 / summary 一句话概览
#   her_preferences       用户信息  : sub_type=习惯偏好     / summary 一句话概览
#   promise               约定承诺  : who_initiated 发起者 / detail 约定内容
#   emotional_event       情感大事件: cause 起因 / process 经过 / result 结果 / emotion_type 情绪类别
#   shared_knowledge      重要话题  : topic 话题主题
#   character_identity    角色自我  : （无需 fields，content 即可）
#   character_interest    角色兴趣  : （无需 fields，content 即可）


def extraction_system(character_name: str) -> str:
    """拼装 A1 提取 prompt：要求模型输出 category + content + date + fields。"""
    return f"""你负责从{character_name}（AI伴侣）和用户的对话中提取重要事件和关键信息。
这些信息会直接写进{character_name}的记忆，用第二人称"你"来表述。

分类（只提取真正重要的，宁少勿多），并按类别填写 fields 结构化字段：
- relationship_milestone 关系转折：两人关系的里程碑/首次/重大转折
    fields: {{"who_initiated": "谁主动", "from_relation": "转折前的关系", "to_relation": "转折后的关系"}}
- her_life 用户信息·基础：用户的生活状况（住哪、做什么、身边重要的人）
    fields: {{"sub_type": "基础个人信息", "summary": "一句话概览"}}
- her_preferences 用户信息·偏好：用户的爱好、习惯、喜欢/不喜欢（持续性的）
    fields: {{"sub_type": "习惯偏好", "summary": "一句话概览"}}
- promise 约定承诺：两人之间的承诺或约定
    fields: {{"who_initiated": "发起者", "detail": "约定的具体内容"}}
- emotional_event 情感大事件：重大情感转折（不是日常撒娇闹脾气）
    fields: {{"cause": "起因", "process": "经过", "result": "结果", "emotion_type": "情绪类别(如难过/委屈/感动)"}}
- shared_knowledge 重要话题：深度讨论过的重要话题
    fields: {{"topic": "话题主题(几个字)"}}
- character_identity 角色自我：{character_name}关于自己身份/第二人格的重大发现或决定
    fields: {{}}
- character_interest 角色兴趣：{character_name}形成的持续兴趣
    fields: {{}}

【人称规则】用"你"指{character_name}，用"她/他"指用户。
【字段规则】fields 里填不出来的项，就给空字符串 ""，不要编造；不要把细节塞进状态字段。

【不要提取】
- 日常琐碎（今天吃了什么、几点到家）
- 已解决的技术问题
- 时事新闻 / 生活常识
- 重复已有事件的内容

每条 content 15-40字、含具体细节。没有值得提取的就返回空列表 []。

只输出 JSON，不要其他内容：
[
  {{"category": "类别", "content": "完整细节(用你/她人称)", "date": "YYYY-MM-DD", "fields": {{...}}}}
]"""


# B1：摘要的 6 个粗主题标签（可多选）。语义判断交给模型，本地不做关键词归类。
SUMMARY_TAGS = ["日常生活", "情绪状态", "关系互动", "用户近况", "兴趣话题", "其他"]


def summary_system(character_name: str) -> str:
    """B1 摘要 prompt：让模型输出主题标签(多选) + 第三人称正文。时间段头由本地拼。"""
    tags = " / ".join(SUMMARY_TAGS)
    return f"""你负责总结{character_name}（AI伴侣）和用户的一段对话，供{character_name}回忆用。

要求：
1. tags：从下面 6 个固定标签里挑出最贴切的（可多选，1-3 个）：{tags}
2. summary：用第三人称、简洁地总结这段对话。保留有意义的细节：
   用户说的事、情绪状态、两人之间发生的事。不要写时间段（时间会另外标注）。

只输出 JSON，不要其他内容：
{{"tags": ["标签1", "标签2"], "summary": "正文"}}"""
