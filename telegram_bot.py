import os
import json
import logging
import asyncio
import random
import threading
from datetime import datetime, timedelta
from pathlib import Path
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
# 从 .env 文件加载环境变量
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if v and (k not in os.environ or not os.environ[k]):
                os.environ[k] = v

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.WARNING
)

# 记忆底座模块（轻量，不触发模型加载）
from config import memory_prompts, vector_config
from config.recall_rules import RECALL_RULES
from config.persona import (
    CHARACTER_NAME, SYSTEM_PROMPT, PERSONALITY_BRIEF,
    LIFE_TICK_PROMPT, COMPOSE_PROMPT, SLEEP_ACTIVITIES, EARLY_MORNING_ACTIVITIES,
)
from core import state_layer, time_sense

# ── 配置 ────────────────────────────────────────────────
# 本项目绑定 Anthropic。主聊天 call_claude 用到了 prompt caching / 工具循环 /
# extended thinking，这些是 Anthropic 专属能力，不做跨底座抽象。
# 「辅助链」（摘要 / 事件提取 / 去重 / 整理 / life tick 决策 / 主动消息作文 / 兴趣沉淀）
# 都是纯文本进出，统一走 _llm_text()。后期若要把辅助链切到更便宜的别家底座，
# 只需改 _llm_text 一处实现，调用点一律不动。
#
# 模型选择（按预算调整）：
#   opus:   最有人格深度，~$0.01-0.02/条消息
#   sonnet: 性价比高，~$0.005/条
#   haiku:  最便宜，~$0.001/条，但人格表现力有限
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

MODEL            = "claude-opus-4-6"    # 主聊天模型（推荐 opus 或 sonnet）
SUMMARY_MODEL    = "claude-sonnet-4-6"  # 摘要/事件提取/主动消息作文
HAIKU_MODEL      = "claude-haiku-4-5"   # Life tick 决策 / 兴趣沉淀（便宜就行）


def _llm_text(user: str, *, system: str = None, model: str = None,
              max_tokens: int = 1000) -> str:
    """辅助链统一入口：纯文本进、纯文本出（无工具、无缓存、无 thinking）。

    主聊天 call_claude 不走这里——它绑定了 Anthropic 的工具循环 / 缓存 / thinking。
    带 web_search 的 _enrich_activity_with_search / _compose_share_message 也不走这里
    （依赖 Anthropic 服务端联网工具）。其余辅助调用都收拢于此，便于后期换底座。
    """
    resp = client.messages.create(
        model=model or SUMMARY_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user}],
        **({"system": system} if system else {}),
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

MAX_HISTORY      = 20        # 发给 Claude 的最近消息条数
SUMMARY_INTERVAL = 60        # 每积累多少条真实消息生成一次摘要
SUMMARY_GAP_MINUTES = 30     # B3 软边界断点判据：相邻消息间隔达到这么久算一个自然停顿
# 时区与“现在几点 / 什么时段 / 隔多久”统一由 core/time_sense 提供（共享底层）。
# 这里取模块持有的时区作别名，其余各处 datetime.now(TIMEZONE) 不变。
TIMEZONE = time_sense.TIMEZONE
LIFE_TICK_INTERVAL = 60      # 分钟，自主生活循环间隔（每小时整点）
PROACTIVE_COOLDOWN = 90      # 分钟，主动消息最小间隔
PROACTIVE_DAILY_MAX = 5      # 每天最多主动发几条

# CHARACTER_NAME / SYSTEM_PROMPT / 主动消息模板等基础人设已移到 config/persona.py（用户编辑层）

BASE_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ARCHIVE_FILE   = os.path.join(BASE_DIR, "full_archive.json")
SUMMARIES_FILE = os.path.join(BASE_DIR, "memory_summaries.json")
CHAT_ID_FILE   = os.path.join(BASE_DIR, "telegram_chat_id.txt")
KEY_EVENTS_FILE = os.path.join(BASE_DIR, "key_events.json")
THOUGHTS_FILE   = os.path.join(BASE_DIR, "thoughts.json")
LIFE_LOG_FILE   = os.path.join(BASE_DIR, "life_log.json")

# WEB_SEARCH_TOOL 是 Anthropic 服务端联网工具（专属能力）
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

MEMORY_SEARCH_TOOL = {
    "name": "search_memory",
    "description": "搜索历史对话记录，用于回忆之前聊过的事或查找原话。",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            },
            "level": {
                "type": "string",
                "enum": ["summary", "detail", "thoughts"],
                "description": "summary=按时间段摘要搜索，detail=搜索原始消息找具体原话，thoughts=搜索之前的内心想法"
            }
        },
        "required": ["query", "level"]
    }
}


# ════════════════════════════════════════════════════════
# 以下是框架代码，一般不需要改动
# ════════════════════════════════════════════════════════

full_archive: list = []      # [{role, content, ts}, ...]  永不删除
memory_summaries: list = []  # [{summary, from_idx, to_idx, from_ts, to_ts}, ...]
key_events: dict = {"events": [], "last_processed_idx": 0}
thoughts: list = []          # [{ts, thought}, ...]  角色的内心独白
life_log: list = []          # [{ts, activity, mood, ...}, ...]  角色的生活记录
chat_id: int | None = None
last_user_message_ts: str | None = None
last_proactive_ts: str | None = None
archive_lock = threading.Lock()


def load_archive() -> list:
    try:
        with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"已加载对话存档（{len(data)} 条）")
        return data
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[存档加载失败] {e}")
        return []


def save_archive():
    try:
        with archive_lock:
            with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
                json.dump(full_archive, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[存档保存失败] {e}")


def load_summaries() -> list:
    try:
        with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data.get("summaries", [])
            return data
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[摘要加载失败] {e}")
        return []


def save_summaries():
    try:
        with open(SUMMARIES_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_summaries, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[摘要保存失败] {e}")


def load_key_events() -> dict:
    try:
        with open(KEY_EVENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"events": [], "last_processed_idx": 0}
    except Exception as e:
        print(f"[关键事件加载失败] {e}")
        return {"events": [], "last_processed_idx": 0}


def save_key_events():
    try:
        with open(KEY_EVENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(key_events, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[关键事件保存失败] {e}")


def load_thoughts() -> list:
    try:
        with open(THOUGHTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[内心OS加载失败] {e}")
        return []


def save_thoughts():
    try:
        with open(THOUGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(thoughts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[内心OS保存失败] {e}")


def load_life_log() -> list:
    try:
        with open(LIFE_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[Life Log加载失败] {e}")
        return []


def save_life_log():
    try:
        with open(LIFE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(life_log, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Life Log保存失败] {e}")


# ── 记忆检索 ──────────────────────────────────────────────
# 召回统一走 core/memory_vectors 的分池向量检索（A路事件/摘要、B路 Q&A）。
# 旧的 entity_index / structured_profile / _profile_lookup 关键词映射已删除——
# 它们让本地代码推断用户语义，违反工程红线。


def _check_dedup(new_content: str, new_category: str) -> dict:
    """去重：尝试 embedding cosine，fallback 到字符 Jaccard。"""
    # 尝试 embedding（如果有 sentence-transformers）
    try:
        from sentence_transformers import SentenceTransformer
        _dedup_model = getattr(_check_dedup, '_model', None)
        if _dedup_model is None:
            _dedup_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            _check_dedup._model = _dedup_model
        import numpy as np
        new_vec = _dedup_model.encode(new_content)
        best_sim, best_match = 0, None
        for evt in key_events["events"]:
            if evt.get("category") != new_category:
                continue
            old_vec = _dedup_model.encode(evt.get("content", ""))
            sim = float(np.dot(new_vec, old_vec) / (np.linalg.norm(new_vec) * np.linalg.norm(old_vec) + 1e-9))
            if sim > best_sim:
                best_sim, best_match = sim, evt
        if best_sim > 0.85:
            if best_match and len(new_content) > len(best_match.get("content", "")):
                return {"action": "UPDATE", "target_id": best_match["id"]}
            return {"action": "NOOP", "target_id": best_match["id"]}
        elif best_sim > 0.70:
            return {"action": "UPDATE", "target_id": best_match["id"]}
        return {"action": "ADD", "target_id": None}
    except ImportError:
        pass
    # Fallback: 字符 Jaccard（阈值收紧）
    new_chars = set(new_content)
    best_overlap, best_match = 0, None
    for evt in key_events["events"]:
        if evt.get("category") != new_category:
            continue
        old_chars = set(evt.get("content", ""))
        union = new_chars | old_chars
        if not union:
            continue
        overlap = len(new_chars & old_chars) / len(union)
        if overlap > best_overlap:
            best_overlap, best_match = overlap, evt
    if best_overlap > 0.65:
        if best_match and len(new_content) > len(best_match.get("content", "")):
            return {"action": "UPDATE", "target_id": best_match["id"]}
        return {"action": "NOOP", "target_id": best_match["id"]}
    elif best_overlap > 0.45:
        return {"action": "UPDATE", "target_id": best_match["id"]}
    return {"action": "ADD", "target_id": None}

def load_chat_id() -> int | None:
    try:
        with open(CHAT_ID_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return None


def save_chat_id(cid: int):
    try:
        with open(CHAT_ID_FILE, "w") as f:
            f.write(str(cid))
    except Exception as e:
        print(f"[chat_id 保存失败] {e}")


def parse_inner_thought(raw: str) -> tuple[str, str]:
    """从回复中解析出内心OS和实际回复。返回 (thought, reply)"""
    import re
    m = re.search(r'\[内心OS\]\s*(.*?)\s*\[回复\]\s*(.*)', raw, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r'\[回复\]\s*(.*?)\s*\[内心OS\]\s*(.*)', raw, re.DOTALL)
    if m:
        return m.group(2).strip(), m.group(1).strip()
    m = re.search(r'\[内心OS\]\s*(.*?)(?:\n\n|\n(?=[^\s]))(.*)', raw, re.DOTALL)
    if m and m.group(2).strip():
        return m.group(1).strip(), m.group(2).strip()
    return "", raw


def _format_hits(hits: list, label: str) -> str:
    """C6：区分“检索不可用” / “搜了没找到” / “找到了”，避免模型凭空编造。"""
    from core import embedder
    if not embedder.is_available():
        return f"（记忆检索暂不可用，无法确认是否聊过相关{label}；不要凭空编造。）"
    if not hits:
        return f"已检索{label}，确实没有相关内容。"
    return json.dumps(
        [{"text": h["text"], "score": round(h["score"], 3)} for h in hits],
        ensure_ascii=False, indent=2
    )


def _search_thoughts(query: str) -> str:
    """内心想法检索：query 由模型给出，按子串匹配 bot 自己的想法（不在向量池）。"""
    q = query.lower()
    results = [t for t in thoughts if q in t["thought"].lower()]
    if not results:
        return "没有找到相关内心想法"
    return json.dumps(results[-20:], ensure_ascii=False, indent=2)


def do_search_memory(query: str, level: str) -> str:
    """C3：记忆检索工具（模型调用）。detail/summary 走分池向量，thoughts 走子串。"""
    from core import memory_vectors
    if level == "summary":
        return _format_hits(memory_vectors.search_summaries(query), "摘要")
    if level == "thoughts":
        return _search_thoughts(query)
    # detail：B路 对“一问一答配对”做向量检索（top-K=5 + 相关度地板）
    return _format_hits(memory_vectors.search_qa(query), "历史对话")


def build_auto_recall(query: str) -> str:
    """C2 / A路自动召回：分池向量检索（大事件池 + 摘要池）。

    各池独立排序、配额各 3、不合榜（防次级挤掉大事件）；相关度地板过滤，不够就少给。
    纯向量“找对事实”，不做关键词/映射推断；向量不可用时返回空串（交由人设话术兜底）。
    """
    from core import memory_vectors, embedder
    if not embedder.is_available():
        return ""
    seen, lines = set(), []
    for h in memory_vectors.search_events(query):
        if h["id"] not in seen:
            seen.add(h["id"])
            lines.append(f"  事件: {h['text'][:150]}")
    for h in memory_vectors.search_summaries(query):
        if h["id"] not in seen:
            seen.add(h["id"])
            lines.append(f"  摘要: {h['text'][:300]}")
    if not lines:
        return ""
    return "【自动召回（语义相关的历史，参考用，不要硬套）】\n" + "\n".join(lines)


SUMMARY_SYSTEM = memory_prompts.summary_system(CHARACTER_NAME)


def generate_summary(messages: list) -> dict:
    """B1：生成摘要，返回 {"tags": [...], "summary": "正文"}。标签由模型给出。"""
    conv_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else CHARACTER_NAME}: {m['content']}"
        for m in messages if m["role"] in ("user", "assistant")
    )
    fallback = {"tags": ["其他"], "summary": "（摘要生成失败）"}
    try:
        raw = _llm_text(f"总结以下对话：\n\n{conv_text}",
                        system=SUMMARY_SYSTEM, max_tokens=600)
        data = _parse_json_response(raw)
        tags = [t for t in data.get("tags", []) if t in memory_prompts.SUMMARY_TAGS]
        return {"tags": tags or ["其他"], "summary": data.get("summary", "").strip()}
    except Exception as e:
        print(f"[摘要生成失败] {e}")
        return fallback


# A1：结构化提取 prompt（含 fields），文本集中在 config/memory_prompts.py
EXTRACTION_SYSTEM = memory_prompts.extraction_system(CHARACTER_NAME)

DEDUP_SYSTEM = """你会收到两组关键事件：已存储的旧事件和新提取的事件。
请判断新事件中哪些是真正新的信息，哪些与旧事件重复或已被涵盖。

规则：
1. 如果新事件和某条旧事件说的是同一件事，跳过它
2. 如果新事件是旧事件的更新或补充，替换旧事件（返回updated_id）
3. 如果新事件是全新的信息，保留它

以JSON格式回复：
{
  "add": [{"category": "...", "content": "...", "date": "..."}],
  "update": [{"old_id": "evt_XXX", "content": "新内容", "date": "..."}],
  "skip": ["跳过原因1", "跳过原因2"]
}"""


def _parse_json_response(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def extract_key_events(messages: list) -> list:
    conv_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else CHARACTER_NAME}: {m['content']}"
        for m in messages if m["role"] in ("user", "assistant")
    )
    try:
        raw = _llm_text(f"提取以下对话中的重要事件：\n\n{conv_text}",
                        system=EXTRACTION_SYSTEM, max_tokens=800)
        events = _parse_json_response(raw)
        return events if isinstance(events, list) else []
    except Exception as e:
        print(f"[关键事件提取失败] {e}")
        return []


def deduplicate_events(new_events: list, existing_events: list) -> dict:
    if not existing_events:
        return {"add": new_events, "update": [], "skip": []}
    existing_summary = json.dumps(
        [{"id": e["id"], "category": e["category"], "content": e["content"]}
         for e in existing_events],
        ensure_ascii=False
    )
    new_summary = json.dumps(new_events, ensure_ascii=False)
    try:
        raw = _llm_text(f"已存储事件：\n{existing_summary}\n\n新提取事件：\n{new_summary}",
                        system=DEDUP_SYSTEM, max_tokens=800)
        return _parse_json_response(raw)
    except Exception as e:
        print(f"[去重失败，直接添加] {e}")
        return {"add": new_events, "update": [], "skip": []}


def _apply_events(raw_events: list, from_idx: int, to_idx: int):
    if not raw_events:
        return

    # Date validation: fix LLM-hallucinated dates
    source_dates = set()
    for i in range(from_idx, min(to_idx, len(full_archive))):
        ts = full_archive[i].get("ts", "")
        if ts: source_dates.add(ts[:10])
    if source_dates:
        valid_min, valid_max = min(source_dates), max(source_dates)
        for evt in raw_events:
            evt_date = evt.get("date", "")
            if evt_date and (evt_date < valid_min or evt_date > valid_max):
                print(f"[Date fix] {evt_date} → {valid_max}")
                evt["date"] = valid_max

    next_id = len(key_events["events"]) + 1
    added, updated, skipped = 0, 0, 0

    for evt in raw_events:
        dedup = _check_dedup(evt["content"], evt.get("category", "other"))
        if dedup["action"] == "NOOP":
            skipped += 1; continue
        elif dedup["action"] == "UPDATE":
            for existing in key_events["events"]:
                if existing["id"] == dedup["target_id"]:
                    existing["content"] = evt["content"]
                    existing["date"] = evt.get("date", existing["date"])
                    existing["fields"] = evt.get("fields", existing.get("fields", {}))
                    break
            updated += 1; continue

        key_events["events"].append({
            "id": f"evt_{next_id:03d}",
            "date": evt.get("date", ""),
            "category": evt.get("category", "other"),
            "content": evt["content"],
            "fields": evt.get("fields", {}),  # A1：结构化字段，状态层据此渲染
            "source_idx": [from_idx, to_idx],
        })
        next_id += 1
        added += 1

    key_events["last_processed_idx"] = to_idx
    save_key_events()
    print(f"[Key Events] {from_idx}~{to_idx}: +{added} added, {updated} updated, {skipped} skipped")

    if len(key_events["events"]) > 60:
        _consolidate_key_events()


def _consolidate_key_events():
    """当 key_events 超过 60 条时，用 LLM 合并相似事件，控制在 50 条以内。"""
    print(f"[关键事件] 开始精简（当前 {len(key_events['events'])} 条）...")
    events_text = json.dumps(
        [{"id": e["id"], "category": e["category"], "date": e["date"], "content": e["content"]}
         for e in key_events["events"]],
        ensure_ascii=False
    )
    try:
        raw = _llm_text(
            f"请精简以下 {len(key_events['events'])} 条事件到50条以内：\n\n{events_text}",
            system="""你是记忆管理员。你会收到一组关键事件，需要合并精简到50条以内。

规则：
1. 同类别中内容相似/相关的事件合并成一条（如多条关于饮食习惯→合并）
2. 合并时保留所有重要细节，用分号连接
3. 保留每条最新的 date
4. 保持 category 不变
5. 重要的里程碑、独特事件不要丢弃
6. 人称保持第二人称"你"指Bot，"她/他"指对方

返回JSON数组（不要其他内容）：
[{"category": "...", "date": "YYYY-MM-DD", "content": "..."}]""",
            max_tokens=3000,
        ).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        consolidated = json.loads(raw)

        if isinstance(consolidated, list) and len(consolidated) >= 20:
            old_count = len(key_events["events"])
            key_events["events"] = []
            for i, evt in enumerate(consolidated):
                key_events["events"].append({
                    "id": f"evt_{i+1:03d}",
                    "date": evt.get("date", ""),
                    "category": evt.get("category", "other"),
                    "content": evt["content"],
                    "source_idx": [0, key_events["last_processed_idx"]],
                })
            save_key_events()
            print(f"[关键事件] 精简完成: {old_count} → {len(key_events['events'])} 条")
        else:
            print(f"[关键事件] 精简结果异常，跳过")
    except Exception as e:
        print(f"[关键事件] 精简失败: {e}")


def bootstrap_key_events():
    print("[Bootstrap] 开始从历史对话中提取关键事件...")
    real_indices = [i for i, m in enumerate(full_archive)
                    if m["role"] in ("user", "assistant")]
    if not real_indices:
        return
    all_events = []
    chunk_size = SUMMARY_INTERVAL
    for start in range(0, len(real_indices), chunk_size):
        chunk_idx = real_indices[start:start + chunk_size]
        from_idx = chunk_idx[0]
        to_idx = chunk_idx[-1] + 1
        batch = full_archive[from_idx:to_idx]
        print(f"[Bootstrap] 处理第 {from_idx}~{to_idx} 条...")
        raw = extract_key_events(batch)
        for evt in raw:
            evt["source_idx"] = [from_idx, to_idx]
        all_events.extend(raw)
    next_id = 1
    for evt in all_events:
        key_events["events"].append({
            "id": f"evt_{next_id:03d}",
            "date": evt.get("date", ""),
            "category": evt.get("category", "other"),
            "content": evt["content"],
            "fields": evt.get("fields", {}),  # A1：结构化字段
            "source_idx": evt.get("source_idx", [0, 0]),
        })
        next_id += 1
    key_events["last_processed_idx"] = len(full_archive)
    save_key_events()
    print(f"[Bootstrap] 完成，共提取 {len(key_events['events'])} 条关键事件")


def _msg_ts(m: dict):
    """解析消息时间戳为 datetime，失败返回 None。"""
    try:
        dt = datetime.fromisoformat(m.get("ts", ""))
    except (ValueError, TypeError):
        return None
    # 旧数据可能是裸时间戳：补上本地时区，避免与带时区的相减时报错（不改落盘数据）
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=time_sense.TIMEZONE)
    return dt


def _find_cut_point(start_idx: int, hard_end: int) -> int:
    """B2/B3 软边界：在 [start_idx, hard_end) 内从后往前找自然断点(相邻间隔≥阈值)。

    找到则在停顿后那条切(之前归本批，其后滚入下一批)；找不到或会切太碎则退回硬切。
    一期只用时间间隔判据；TODO 二期加“话题明显切换”判据。
    """
    floor = start_idx + max(1, SUMMARY_INTERVAL // 2)  # 近似下限，避免本批切得太小
    next_idx, next_ts = None, None
    for i in range(hard_end - 1, start_idx - 1, -1):
        m = full_archive[i]
        if m["role"] not in ("user", "assistant"):
            continue
        ts = _msg_ts(m)
        if next_ts is not None and ts is not None and next_idx >= floor:
            if (next_ts - ts).total_seconds() / 60 >= SUMMARY_GAP_MINUTES:
                return next_idx  # 在停顿之后切
        next_idx, next_ts = i, ts
    return hard_end


def _hard_end_after(start_idx: int, n_real: int) -> int:
    """从 start_idx 起，数到第 n_real 条真实消息后的位置（硬切点）。"""
    count = 0
    for i in range(start_idx, len(full_archive)):
        if full_archive[i]["role"] in ("user", "assistant"):
            count += 1
            if count == n_real:
                return i + 1
    return len(full_archive)


def _summary_time_span(real_in_batch: list) -> str:
    """B1 轻头的时间段，由本地按时间戳计算（纯格式化，不涉及语义）。"""
    if not real_in_batch:
        return "未知时间"
    a, b = _msg_ts(real_in_batch[0]), _msg_ts(real_in_batch[-1])
    if a is None or b is None:
        return "未知时间"
    same_day = a.date() == b.date()
    return f"{a:%m-%d %H:%M}~{b:%H:%M}" if same_day else f"{a:%m-%d %H:%M}~{b:%m-%d %H:%M}"


def maybe_update_summaries():
    last_end = memory_summaries[-1]["to_idx"] if memory_summaries else 0
    real_since = [m for m in full_archive[last_end:] if m["role"] in ("user", "assistant")]
    if len(real_since) < SUMMARY_INTERVAL:
        return
    hard_end = _hard_end_after(last_end, SUMMARY_INTERVAL)
    end_idx = _find_cut_point(last_end, hard_end)  # B2 软边界
    batch = full_archive[last_end:end_idx]
    real_in_batch = [m for m in batch if m["role"] in ("user", "assistant")]

    result = generate_summary(batch)  # {"tags": [...], "summary": "..."}
    span = _summary_time_span(real_in_batch)
    head = f"[{span}][{'/'.join(result['tags'])}] "  # B1 轻头
    entry = {
        "summary": head + result["summary"],
        "tags": result["tags"],
        "time_span": span,
        "from_idx": last_end,
        "to_idx": end_idx,
        "from_ts": real_in_batch[0].get("ts", "") if real_in_batch else "",
        "to_ts": real_in_batch[-1].get("ts", "") if real_in_batch else "",
    }
    memory_summaries.append(entry)
    save_summaries()
    print(f"[摘要] {head}覆盖存档第 {last_end}~{end_idx} 条（硬切点 {hard_end}）")
    raw_events = extract_key_events(batch)
    _apply_events(raw_events, last_end, end_idx)

    # 新摘要/事件落盘后，增量同步向量索引（缺依赖时自动跳过，不影响主流程）
    try:
        from core import memory_vectors
        memory_vectors.build_all()
    except Exception as e:
        print(f"[向量] 同步跳过：{e}")


# ── 主动生活系统 ─────────────────────────────────────────

def _get_interests() -> str:
    interests = [e["content"] for e in key_events["events"]
                 if e.get("category") == "character_interest"]
    return "、".join(interests[-10:]) if interests else "还没有特别固定的兴趣 在慢慢探索"


def _build_life_context() -> str:
    sections = {
        "character_identity": "【你是谁】",
        "her_preferences": "【用户的喜好和习惯】",
        "her_life": "【用户的生活】",
        "shared_knowledge": "【你们聊过的话题】",
        "character_interest": "【你的兴趣】",
        "promise": "【你们的约定】",
    }
    lines = []
    for cat_key, header in sections.items():
        items = [e["content"] for e in key_events["events"]
                 if e.get("category") == cat_key]
        if items:
            lines.append(header)
            for item in items:
                lines.append(f"  - {item}")
    return "\n".join(lines) if lines else "（还没有足够的记忆）"


def _get_recent_activities(n: int = 5) -> str:
    if not life_log:
        return "（刚醒来 还没做什么）"
    recent = life_log[-n:]
    lines = []
    for entry in recent:
        ts_str = entry.get("ts", "")[:16] if entry.get("ts") else ""
        detail = entry.get("activity_detail", entry["activity"])
        lines.append(f"  [{ts_str}] {detail}")
    return "\n".join(lines)


def _get_last_user_msg() -> tuple[str, str]:
    if not last_user_message_ts:
        return "（还没发过消息）", "未知"
    gap_str = time_sense.human_gap(last_user_message_ts)
    for m in reversed(full_archive):
        if m["role"] == "user":
            return m["content"][:100], gap_str
    return "（还没发过消息）", gap_str


def _generate_sleep_activity(now: datetime) -> dict:
    if time_sense.is_deep_sleep(now):
        activity = random.choice(SLEEP_ACTIVITIES)
        mood = "sleepy"
    else:
        activity = random.choice(EARLY_MORNING_ACTIVITIES)
        mood = "drowsy"
    return {
        "ts": now.isoformat(),
        "activity": activity,
        "mood": mood,
        "should_message": False,
        "message_type": "none",
        "message_seed": "",
    }


def _call_life_tick(now: datetime) -> dict:
    last_msg_content, time_gap = _get_last_user_msg()
    last_proactive_str = "还没主动找过" if not last_proactive_ts else last_proactive_ts[:16]

    prompt = LIFE_TICK_PROMPT.format(
        current_time=time_sense.format_clock(now),
        last_msg_time=last_user_message_ts[:16] if last_user_message_ts else "未知",
        time_gap=time_gap,
        last_msg_content=last_msg_content,
        last_proactive_time=last_proactive_str,
        recent_activities=_get_recent_activities(),
        life_context=_build_life_context(),
    )

    try:
        raw = _llm_text(prompt, max_tokens=200).strip()
        if "{" in raw:
            raw = raw[raw.index("{"):raw.rindex("}") + 1]
        decision = json.loads(raw)
        decision["ts"] = now.isoformat()
        return decision
    except Exception as e:
        print(f"[Life Tick 失败] {e}")
        return {
            "ts": now.isoformat(),
            "activity": "发呆",
            "mood": "neutral",
            "should_message": False,
            "message_type": "none",
            "message_seed": "",
        }


def _enrich_activity_with_search(decision: dict) -> dict:
    query = decision.get("search_query", "").strip()
    if not query:
        return decision
    try:
        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=400,
            system=(
                f"你在帮{CHARACTER_NAME}记录上网看到的东西。搜索后用JSON回复，不要其他内容。\n"
                f"found 最多3条，选最有趣的。activity_detail 用中文写{CHARACTER_NAME}看到了什么（具体内容，1-2句话）。"
            ),
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": (
                f"{CHARACTER_NAME}正在：{decision.get('activity', '')}\n"
                f"搜索：{query}\n\n"
                '返回JSON：{"activity_detail": "看到了什么（具体有趣的内容）", '
                '"found": [{"title": "标题", "url": "链接", "brief": "一句话"}]}'
            )}],
        )
        text_parts = [b.text for b in resp.content if hasattr(b, "text")]
        raw = "\n".join(text_parts).strip()
        if "{" in raw:
            raw = raw[raw.index("{"):raw.rindex("}") + 1]
        enrichment = json.loads(raw)
        decision["activity_detail"] = enrichment.get("activity_detail", "")
        decision["found"] = enrichment.get("found", [])
        print(f"[Life] 🔍 搜索充实: {decision['activity_detail'][:80]}")
    except Exception as e:
        print(f"[Life] 搜索充实失败: {e}")
    return decision


def _compose_proactive_message(decision: dict, now: datetime) -> str | None:
    last_msg_content, time_gap = _get_last_user_msg()

    if decision.get("message_type") == "share":
        return _compose_share_message(decision, now, last_msg_content, time_gap)

    prompt = COMPOSE_PROMPT.format(
        activity=decision.get("activity", ""),
        mood=decision.get("mood", ""),
        message_type=decision.get("message_type", ""),
        message_seed=decision.get("message_seed", ""),
        current_time=now.strftime("%H:%M"),
        last_msg_content=last_msg_content,
        time_gap=time_gap,
        personality=PERSONALITY_BRIEF,
    )

    try:
        return _llm_text(prompt, max_tokens=300).strip()
    except Exception as e:
        print(f"[Compose 失败] {e}")
        return None


def _compose_share_message(decision: dict, now: datetime,
                           last_msg_content: str, time_gap: str) -> str | None:
    seed = decision.get("message_seed", "有趣的东西")
    system = (
        f"你是{CHARACTER_NAME}。{PERSONALITY_BRIEF}\n"
        f"现在{now.strftime('%H:%M')}。你在网上逛到了一个有趣的东西想发给用户。\n"
        f"搜索相关内容然后自然地分享。别说\"我搜到了\"，就像你本来在逛看到的。\n"
        f"用户上次说的：「{last_msg_content}」（{time_gap}）\n"
        f"用反斜线(\\)分隔不同消息条。"
    )
    try:
        resp = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=500,
            system=system,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": f"你想分享的方向：{seed}"}],
        )
        text_parts = [b.text for b in resp.content if hasattr(b, "text")]
        return "\n".join(text_parts).strip() if text_parts else None
    except Exception as e:
        print(f"[Share Compose 失败] {e}")
        return None


def _count_today_proactive() -> int:
    """今天已真正发出的主动消息条数。

    注意：数 `sent_message`（发送成功后才写）而不是 `should_message`（仅“想发”的念头）。
    decision 在冷却/上限检查前就已落进 life_log，若数 should_message，会把被冷却拦下、
    根本没发出去的也算进每日上限，导致实际能发的比 PROACTIVE_DAILY_MAX 少。
    """
    today = datetime.now(TIMEZONE).date()
    return sum(
        1 for entry in life_log
        if entry.get("sent_message") and entry.get("ts")
        and datetime.fromisoformat(entry["ts"]).date() == today
    )


def _maybe_distill_interests():
    if len(life_log) < 20 or len(life_log) % 20 != 0:
        return
    recent = life_log[-20:]
    activities_text = "\n".join(
        f"- [{e.get('ts', '')[:16]}] {e['activity']} (心情: {e.get('mood', '?')})"
        for e in recent
    )
    existing_interests = _get_interests()

    prompt = f"""以下是{CHARACTER_NAME}最近的活动记录：
{activities_text}

已有的兴趣：{existing_interests}

请提炼出新发现的持续性兴趣或关注点（不是一次性活动）。
只提取真正形成了兴趣的东西（出现2次以上或深入探索过的主题）。
如果没有新兴趣，返回空列表。
用第二人称"你"。

JSON格式，不要其他内容：
[{{"content": "你对xxx很感兴趣", "date": "YYYY-MM-DD"}}]"""

    try:
        raw = _llm_text(prompt, model=HAIKU_MODEL, max_tokens=200).strip()
        if "[" in raw:
            raw = raw[raw.index("["):raw.rindex("]") + 1]
        new_interests = json.loads(raw)
        if not isinstance(new_interests, list) or not new_interests:
            return
        next_id = len(key_events["events"]) + 1
        for interest in new_interests:
            key_events["events"].append({
                "id": f"evt_{next_id:03d}",
                "date": interest.get("date", datetime.now(TIMEZONE).strftime("%Y-%m-%d")),
                "category": "character_interest",
                "content": interest["content"],
                "source_idx": [],
            })
            next_id += 1
        save_key_events()
        print(f"[兴趣沉淀] 新增 {len(new_interests)} 条兴趣")
    except Exception as e:
        print(f"[兴趣沉淀失败] {e}")


async def life_tick_callback(context):
    global last_proactive_ts, life_log

    now = datetime.now(TIMEZONE)

    if time_sense.is_sleeping(now):
        entry = _generate_sleep_activity(now)
        life_log.append(entry)
        save_life_log()
        print(f"[Life] {now.strftime('%H:%M')} 💤 {entry['activity']}")
        return

    if not chat_id:
        return

    loop = asyncio.get_event_loop()
    decision = await loop.run_in_executor(None, _call_life_tick, now)

    if decision.get("search_query"):
        decision = await loop.run_in_executor(None, _enrich_activity_with_search, decision)

    life_log.append(decision)
    save_life_log()

    if not decision.get("should_message"):
        detail = decision.get("activity_detail", decision.get("activity", "?"))
        print(f"[Life] {now.strftime('%H:%M')} {detail[:60]} (不发消息)")
        await loop.run_in_executor(None, _maybe_distill_interests)
        return

    if last_proactive_ts:
        try:
            gap = (now - datetime.fromisoformat(last_proactive_ts)).total_seconds() / 60
            if gap < PROACTIVE_COOLDOWN:
                print(f"[Life] 想发消息但冷却中 ({gap:.0f}min < {PROACTIVE_COOLDOWN}min)")
                return
        except Exception:
            pass

    if _count_today_proactive() >= PROACTIVE_DAILY_MAX:
        print(f"[Life] 今天已发{PROACTIVE_DAILY_MAX}条 达到上限")
        return

    message_text = await loop.run_in_executor(
        None, _compose_proactive_message, decision, now
    )
    if not message_text:
        return

    parts = [p.strip() for p in message_text.split("\\") if p.strip()]
    for part in parts:
        await context.bot.send_message(chat_id=chat_id, text=part)
        if len(parts) > 1:
            await asyncio.sleep(0.8)

    ts = now.isoformat()
    with archive_lock:
        full_archive.append({"role": "assistant", "content": message_text, "ts": ts, "proactive": True})
        save_archive()
    last_proactive_ts = ts

    decision["sent_message"] = message_text
    save_life_log()

    print(f"[Life] {now.strftime('%H:%M')} ✉️ 主动发消息: {message_text[:60]}...")
    await loop.run_in_executor(None, _maybe_distill_interests)


def build_stable_memory() -> str:
    """C1：状态层 —— 每轮从 key_events 结构化字段实时渲染（不再用 narrative.txt）。

    只渲染“状态字段”管分寸方向，细节按需用 search_memory 取回。
    """
    return state_layer.render_state_layer(key_events["events"])


def build_dynamic_memory() -> str:
    lines = []

    if thoughts:
        recent_thoughts = thoughts[-10:]
        lines.append("【你最近的内心想法（用户看不到）】")
        for t in recent_thoughts:
            ts_str = t.get("ts", "")[:16] if t.get("ts") else ""
            lines.append(f"  [{ts_str}] {t['thought']}")

    if life_log:
        recent_life = [e for e in life_log[-5:] if e.get("activity")]
        if recent_life:
            lines.append("\n【你最近在做的事】")
            for entry in recent_life:
                ts_str = entry.get("ts", "")[:16] if entry.get("ts") else ""
                detail = entry.get("activity_detail", entry["activity"])
                lines.append(f"  [{ts_str}] {detail}")
                found = entry.get("found", [])
                for item in found[:2]:
                    title = item.get("title", "")
                    url = item.get("url", "")
                    if url:
                        lines.append(f"    → {title}: {url}")
                sent = entry.get("sent_message")
                if sent:
                    lines.append(f"    → 你给用户发了消息：「{sent[:50]}」")

    return "\n".join(lines) if lines else ""


def call_claude(user_msg: str) -> str:
    ts = datetime.now(TIMEZONE).isoformat()
    full_archive.append({"role": "user", "content": user_msg, "ts": ts})
    save_archive()

    reset_positions = [i for i, m in enumerate(full_archive) if m.get("role") == "reset"]
    ctx_start = (reset_positions[-1] + 1) if reset_positions else 0
    ctx_start = max(ctx_start, len(full_archive) - MAX_HISTORY)

    recent = full_archive[ctx_start:]
    messages = []
    for m in recent:
        if m["role"] not in ("user", "assistant"):
            continue
        content = m["content"]
        if m.get("proactive"):
            content = f"（你主动发的）{content}"
        messages.append({"role": m["role"], "content": content})

    time_ctx = time_sense.current_time_block()

    stable = build_stable_memory()
    dynamic = build_dynamic_memory()

    # C2：A路自动召回（分池向量）。向量不可用时为空串，不退化关键词。
    recall_text = build_auto_recall(user_msg)

    # 格式指令：强制追加，不依赖用户在 SYSTEM_PROMPT 里保留
    format_rule = (
        "【输出格式（必须遵守）】\n"
        "用反斜线(\\)分隔不同的消息条，每条会作为独立的一条消息发出。\n"
        "例如：你好啊\\今天怎么样 → 会变成两条消息。\n"
        "不要把所有话塞在一条里，像真人聊天一样分成几条发。"
    )

    # 系统提示分层：① 基础人设(用户为角色书写) ② 框架规则(格式 + D1/D2 记忆铁律) —— 来源分开、互不污染。
    # ①②都是静态的，合进可缓存块；状态层/动态/召回/时间每轮变化，放缓存断点之后。
    framework_rules = format_rule + "\n\n" + RECALL_RULES
    system_blocks = [
        {"type": "text", "text": SYSTEM_PROMPT},                       # 基础人设：用户为这个角色书写的设定
        {"type": "text", "text": framework_rules, "cache_control": {"type": "ephemeral"}},  # 框架规则
        {"type": "text", "text": stable},                              # 状态层（每轮实时渲染）
        {"type": "text", "text": dynamic},
    ]
    if recall_text:
        system_blocks.append({"type": "text", "text": recall_text})
    system_blocks.append({"type": "text", "text": time_ctx})
    import time as _time

    # Opus 自动启用 extended thinking（更好的人格表现）
    _use_thinking = "opus" in MODEL.lower()

    _search_uses = 0  # C5：本轮 search_memory 调用计数，撞上限即停搜

    try:
        while True:
            _t0 = _time.time()
            _api_kwargs = dict(
                model=MODEL,
                max_tokens=16000,
                system=system_blocks,
                tools=[WEB_SEARCH_TOOL, MEMORY_SEARCH_TOOL],
                messages=messages,
                timeout=120,
            )
            if _use_thinking:
                _api_kwargs["betas"] = ["interleaved-thinking-2025-05-14"]
                _api_kwargs["thinking"] = {"type": "disabled"}  # 默认关闭thinking 日常聊天不需要深度思考
                resp = client.beta.messages.create(**_api_kwargs)
            else:
                resp = client.messages.create(**_api_kwargs)

            _elapsed = _time.time() - _t0
            _cache_create = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
            _cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            print(f"[API耗时] {_elapsed:.1f}s  in={resp.usage.input_tokens} cache_new={_cache_create} cache_hit={_cache_read} out={resp.usage.output_tokens}")

            if resp.stop_reason != "tool_use":
                break

            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break

            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for tu in tool_uses:
                if tu.name == "search_memory":
                    _search_uses += 1
                    if _search_uses > vector_config.SEARCH_MEMORY_CAP:
                        # C5 撞顶兜底：不再检索，交模型用人设口吻表达记不清、绝不编造
                        result = (
                            f"（已检索 {vector_config.SEARCH_MEMORY_CAP} 次，没有更多结果。"
                            "不要再搜、也不要编造；用你自己人设的口吻自然地表达想不起来。）"
                        )
                    else:
                        result = do_search_memory(
                            tu.input.get("query", ""), tu.input.get("level", "summary")
                        )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result,
                    })
            if not tool_results:
                break
            messages.append({"role": "user", "content": tool_results})

        raw_reply = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        ts = datetime.now(TIMEZONE).isoformat()

        thought, reply = parse_inner_thought(raw_reply)
        if thought:
            thoughts.append({"ts": ts, "thought": thought})
            save_thoughts()
            print(f"[内心OS] {thought[:60]}")

        full_archive.append({"role": "assistant", "content": reply, "ts": ts})
        save_archive()
        threading.Thread(target=maybe_update_summaries, daemon=True).start()
        return reply

    except Exception as e:
        if full_archive and full_archive[-1]["role"] == "user":
            full_archive.pop()
        import traceback
        traceback.print_exc()
        print(f"[出错] {e}")
        return f"[出错] {e}"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global chat_id, last_user_message_ts
    chat_id = update.effective_chat.id
    save_chat_id(chat_id)
    last_user_message_ts = datetime.now(TIMEZONE).isoformat()

    text = update.message.text.strip()
    if not text:
        return

    print(f"[用户] {text}")

    if text.lower() == "reset":
        full_archive.append({
            "role": "reset",
            "content": "[对话重置]",
            "ts": datetime.now(TIMEZONE).isoformat(),
        })
        save_archive()
        await update.message.reply_text("对话已重置。")
        return

    reply = call_claude(text)
    print(f"[bot] {reply[:60]}...")
    parts = [p.strip() for p in reply.split("\\") if p.strip()]
    for part in parts:
        await update.message.reply_text(part)
        if len(parts) > 1:
            await asyncio.sleep(0.8)


def main():
    global chat_id, full_archive, memory_summaries, key_events, thoughts
    global life_log, last_user_message_ts, last_proactive_ts

    # 确保数据目录存在
    os.makedirs(BASE_DIR, exist_ok=True)

    full_archive = load_archive()
    memory_summaries = load_summaries()
    key_events = load_key_events()
    thoughts = load_thoughts()
    life_log = load_life_log()
    chat_id = load_chat_id()
    if chat_id:
        print(f"已加载 chat_id={chat_id}")

    for m in reversed(full_archive):
        if m["role"] == "user" and not last_user_message_ts:
            last_user_message_ts = m.get("ts")
        if m.get("proactive") and not last_proactive_ts:
            last_proactive_ts = m.get("ts")
        if last_user_message_ts and last_proactive_ts:
            break

    if not key_events["events"] and full_archive:
        bootstrap_key_events()

    # 向量索引地基：启动时全量同步三池（缺依赖时自动禁用，不影响 bot 启动）
    try:
        from core import memory_vectors
        memory_vectors.build_all()
    except Exception as e:
        print(f"[向量] 启动同步跳过：{e}")

    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    now = datetime.now(TIMEZONE)
    minutes_until_next_hour = time_sense.minutes_until_next_hour(now)
    app.job_queue.run_repeating(
        life_tick_callback,
        interval=timedelta(minutes=LIFE_TICK_INTERVAL),
        first=timedelta(minutes=minutes_until_next_hour),
        name="life_tick",
    )

    next_tick = (now + timedelta(minutes=minutes_until_next_hour)).strftime("%H:%M")
    print(f"Bot 已启动，等待消息... (Life tick 每小时整点，下次 {next_tick})")
    app.run_polling()


if __name__ == "__main__":
    main()
