"""A3：把三层记忆挂上向量索引。

从 data 目录读取源数据（key_events / summaries / full_archive），
分别同步到三个向量池，并对外提供分池检索。
源数据由对话主流程写入，本模块只做“向量化 + 检索”，不改写记忆内容、不推断语义。

对外接口：
  build_all()         —— 全量同步三池（增量编码，可在启动/写入后调用）
  search_events()     —— A路 大事件池检索
  search_summaries()  —— A路 摘要池检索
  search_qa()         —— B路 存档 Q&A 配对检索
"""
import json

from config import vector_config as cfg
from core import embedder
from core.vector_index import VectorIndex

# 三个池各自独立持久化、独立排序、不合榜
_events = VectorIndex(cfg.POOL_EVENTS, cfg.VECTOR_DIR)
_summaries = VectorIndex(cfg.POOL_SUMMARIES, cfg.VECTOR_DIR)
_qa = VectorIndex(cfg.POOL_QA, cfg.VECTOR_DIR)


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# ── 从源数据构建 (id, text) 全集 ──────────────────────────
def _build_event_items():
    """大事件池：每条 key_event 取其 content 做向量文本。"""
    data = _read_json(cfg.KEY_EVENTS_FILE, {"events": []})
    return [(e["id"], e.get("content", ""))
            for e in data.get("events", []) if e.get("content")]


def _build_summary_items():
    """摘要池：每条摘要正文一个单元，id 用覆盖区间标识。"""
    data = _read_json(cfg.SUMMARIES_FILE, [])
    items = data.get("summaries", []) if isinstance(data, dict) else data
    out = []
    for s in items:
        text = s.get("summary", "")
        if text:
            out.append((f"sum_{s.get('from_idx', 0)}_{s.get('to_idx', 0)}", text))
    return out


def _pair_text(archive, i):
    """把第 i 条用户消息与其后最近的非主动回复配成一问一答文本。"""
    user = archive[i].get("content", "")
    if not user:
        return ""
    for j in range(i + 1, min(i + 4, len(archive))):
        nxt = archive[j]
        if nxt.get("role") == "assistant" and not nxt.get("proactive"):
            return f"用户：{user}\nAI：{nxt.get('content', '')}"
    return f"用户：{user}"


def _build_qa_items():
    """存档 Q&A 池：一问一答配对；主动消息单独成单元。"""
    archive = _read_json(cfg.ARCHIVE_FILE, [])
    out = []
    for i, m in enumerate(archive):
        if m.get("role") == "assistant" and m.get("proactive"):
            out.append((f"pro_{i}", f"（主动消息）{m.get('content', '')}"))
        elif m.get("role") == "user":
            text = _pair_text(archive, i)
            if text:
                out.append((f"qa_{i}", text))
    return out


# ── 对外接口 ──────────────────────────────────────────────
def build_all():
    """全量同步三个池（增量：未变更的不重新编码）。返回是否成功。"""
    if not embedder.is_available():
        print("[向量] 不可用，跳过索引构建")
        return False
    _events.sync(_build_event_items())
    _summaries.sync(_build_summary_items())
    _qa.sync(_build_qa_items())
    print(f"[向量] 索引就绪：事件{len(_events.ids)} 摘要{len(_summaries.ids)} QA{len(_qa.ids)}")
    return True


def search_events(query, top_k=cfg.EVENTS_TOP_K, floor=cfg.RELEVANCE_FLOOR):
    return _events.search(query, top_k, floor)


def search_summaries(query, top_k=cfg.SUMMARIES_TOP_K, floor=cfg.RELEVANCE_FLOOR):
    return _summaries.search(query, top_k, floor)


def search_qa(query, top_k=cfg.QA_TOP_K, floor=cfg.RELEVANCE_FLOOR):
    return _qa.search(query, top_k, floor)


if __name__ == "__main__":
    # 手动重建索引：python -m core.memory_vectors
    build_all()
