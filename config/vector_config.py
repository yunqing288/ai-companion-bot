"""向量层配置：模型名、持久化目录、三池名称、召回参数。

所有可调项集中在此，便于后续把散落在 telegram_bot.py 里的常量逐步迁进 config。
模型为本地 sentence-transformers，无需 API key；如需替换模型，用环境变量覆盖。
"""
import os

# 项目根目录（config 的上一级）
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 嵌入模型 ──────────────────────────────────────────────
# 多语言模型，适合中文。可用环境变量 EMBED_MODEL 覆盖。
EMBED_MODEL = os.environ.get("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# ── 数据与持久化路径 ──────────────────────────────────────
DATA_DIR = os.path.join(_BASE, "data")
VECTOR_DIR = os.path.join(DATA_DIR, "vectors")           # 向量索引落盘目录
KEY_EVENTS_FILE = os.path.join(DATA_DIR, "key_events.json")
SUMMARIES_FILE = os.path.join(DATA_DIR, "memory_summaries.json")
ARCHIVE_FILE = os.path.join(DATA_DIR, "full_archive.json")

# ── 三个向量池（对应三层金字塔）────────────────────────────
POOL_EVENTS = "events"        # 大事件池（Tier1 key_events）
POOL_SUMMARIES = "summaries"  # 摘要池（Tier2 摘要正文）
POOL_QA = "qa"                # 存档 Q&A 配对池（Tier3）

# ── 召回参数（与 recall-redesign-plan 对齐）────────────────
EVENTS_TOP_K = 3        # A路 大事件池配额（各池独立，不合榜）
SUMMARIES_TOP_K = 3     # A路 摘要池配额
QA_TOP_K = 5            # B路 Q&A top-K
RELEVANCE_FLOOR = 0.30  # 相关度地板：低于此分不返回（“不够就少给，找不到一律不编”）

# C5：单轮回复里 search_memory 的最大调用次数（B路兜底闸）。
# 撞上限后不再执行检索，交模型用符合人设的方式表达记不清，绝不编造。
SEARCH_MEMORY_CAP = 2
