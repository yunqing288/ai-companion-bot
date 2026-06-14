"""状态层渲染（C1）：从 key_events 的结构化字段实时渲染“状态视图”。

状态层只读“状态字段”、管常驻分寸方向，不读细节、不参与召回；
细节层（完整 content）按需经向量召回取回。每轮实时渲染，不落盘 narrative。

各类别在状态层只渲染什么（与 recall-redesign-plan 对齐）：
  关系转折   → from→to 关系
  用户信息   → 一句话概览（summary）
  约定承诺   → 仅“存在约定”，不读内容
  情感大事件 → 仅情绪类别
  重要话题   → 仅话题主题
  角色兴趣/自我 → 完整 content 详渲染
"""


def _group(events):
    """按 category 分组。"""
    buckets = {}
    for evt in events:
        buckets.setdefault(evt.get("category", "other"), []).append(evt)
    return buckets


def _field(evt, key, default=""):
    return (evt.get("fields") or {}).get(key, default)


def _render_user_info(buckets):
    """用户信息：her_life + her_preferences，状态读一句话概览。"""
    items = buckets.get("her_life", []) + buckets.get("her_preferences", [])
    lines = []
    for evt in items:
        summary = _field(evt, "summary") or evt.get("content", "")
        if summary:
            lines.append(f"  - {summary}")
    return "【用户信息】\n" + "\n".join(lines) if lines else ""


def _render_relationship(buckets):
    """关系转折：状态只读 from→to。"""
    lines = []
    for evt in buckets.get("relationship_milestone", []):
        frm, to = _field(evt, "from_relation"), _field(evt, "to_relation")
        if frm and to:
            lines.append(f"  - {frm} → {to}")
        elif to:
            lines.append(f"  - 现在：{to}")
    return "【关系】\n" + "\n".join(lines) if lines else ""


def _render_promise(buckets):
    """约定承诺：只说存在、不展开内容。"""
    n = len(buckets.get("promise", []))
    if not n:
        return ""
    return f"【约定】存在 {n} 条约定（内容不展开，需要时用 search_memory 取）"


def _render_emotion(buckets):
    """情感大事件：状态只读情绪类别。"""
    types = [_field(evt, "emotion_type") for evt in buckets.get("emotional_event", [])]
    types = [t for t in types if t]
    return "【情绪事件】" + "、".join(types) if types else ""


def _render_topic(buckets):
    """重要话题：状态只读话题主题。"""
    topics = [_field(evt, "topic") or evt.get("content", "")[:10]
              for evt in buckets.get("shared_knowledge", [])]
    topics = [t for t in topics if t]
    return "【聊过的重要话题】" + "、".join(topics) if topics else ""


def _render_character(buckets):
    """角色自我 + 角色兴趣：完整详渲染。"""
    lines = []
    for evt in buckets.get("character_identity", []):
        if evt.get("content"):
            lines.append(f"  - {evt['content']}")
    interests = [e.get("content", "") for e in buckets.get("character_interest", [])]
    interests = [i for i in interests if i]
    head = "【你是谁 / 你的兴趣】" if lines or interests else ""
    if interests:
        lines.append("  兴趣：" + "、".join(interests))
    return head + "\n" + "\n".join(lines) if head else ""


def render_state_layer(events) -> str:
    """把 key_events 渲染成状态层文本；无内容返回空串。"""
    if not events:
        return ""
    buckets = _group(events)
    sections = [
        _render_user_info(buckets),
        _render_relationship(buckets),
        _render_promise(buckets),
        _render_emotion(buckets),
        _render_topic(buckets),
        _render_character(buckets),
    ]
    body = "\n".join(s for s in sections if s)
    if not body:
        return ""
    return "【记忆状态层（方向把控，细节用 search_memory 取）】\n" + body
