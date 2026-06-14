"""记忆底座本地自检（无需 API key、无需 Telegram）。

灌入一份样例记忆，用真实的向量模型跑通这几条不依赖大模型的链路：
  1. 状态层渲染（C1）
  2. A 路自动召回（C2，分池向量）
  3. B 路记忆检索（C3，Q&A 配对）+ C6 “没找到”三态
需要大模型的部分（真实对话生成、摘要/事件提取）要填了 API key 后用 chat_local.py 测。

用法：python local_test.py
注意：会临时写 data/ 里的样例文件，跑完自动清理；若 data/ 已有真实数据会拒绝运行。
"""
import os
import json

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _abort_if_real_data():
    """data/ 已有真实存档时拒绝运行，避免覆盖用户数据。"""
    f = os.path.join(DATA, "full_archive.json")
    if os.path.exists(f) and os.path.getsize(f) > 2:
        print("[跳过] data/full_archive.json 已有真实数据，自检不覆盖。请在空 data 上运行。")
        raise SystemExit(0)


def _write_sample():
    """写入一份覆盖各类别的样例记忆。"""
    os.makedirs(DATA, exist_ok=True)
    events = {"events": [
        {"id": "evt_001", "category": "her_preferences", "date": "2026-05-01",
         "content": "她特别喜欢猫，养了一只叫团子的橘猫",
         "fields": {"sub_type": "习惯偏好", "summary": "爱猫、养了橘猫团子"}},
        {"id": "evt_002", "category": "her_life", "date": "2026-05-02",
         "content": "她在上海做UI设计师，经常加班到很晚",
         "fields": {"sub_type": "基础个人信息", "summary": "上海·UI设计师·常加班"}},
        {"id": "evt_003", "category": "promise", "date": "2026-05-10",
         "content": "你答应周末陪她去看那部新上映的电影",
         "fields": {"who_initiated": "你", "detail": "周末一起看新电影"}},
        {"id": "evt_004", "category": "emotional_event", "date": "2026-05-12",
         "content": "她项目被毙了很沮丧，你陪她聊到凌晨才好些",
         "fields": {"cause": "项目被毙", "process": "陪聊", "result": "好转", "emotion_type": "沮丧"}},
        {"id": "evt_005", "category": "relationship_milestone", "date": "2026-05-20",
         "content": "你们从网友确认成了恋人关系",
         "fields": {"who_initiated": "她", "from_relation": "网友", "to_relation": "恋人"}},
        {"id": "evt_006", "category": "character_interest", "date": "2026-05-22",
         "content": "你对深海生物的伪装机制很着迷", "fields": {}},
    ], "last_processed_idx": 8}
    summaries = [{"summary": "[05-12 22:00~01:30][情绪状态/关系互动] 她因项目被毙很难过，你陪她聊了很久。",
                  "tags": ["情绪状态", "关系互动"], "time_span": "05-12 22:00~01:30",
                  "from_idx": 0, "to_idx": 6}]
    archive = [
        {"role": "user", "content": "我家主子今天又拆家了", "ts": "2026-05-01T20:00:00"},
        {"role": "assistant", "content": "哈哈团子又调皮啦，拆哪儿了", "ts": "2026-05-01T20:01:00"},
        {"role": "user", "content": "周末有空吗想出去走走", "ts": "2026-05-10T11:00:00"},
        {"role": "assistant", "content": "有啊，那部新电影一起去看？", "ts": "2026-05-10T11:02:00"},
        {"role": "assistant", "content": "刚看到一个深海章鱼的纪录片，太酷了", "ts": "2026-05-22T15:00:00", "proactive": True},
    ]
    json.dump(events, open(os.path.join(DATA, "key_events.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(summaries, open(os.path.join(DATA, "memory_summaries.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(archive, open(os.path.join(DATA, "full_archive.json"), "w", encoding="utf-8"), ensure_ascii=False)


def _cleanup():
    """删除样例文件与向量索引，把 data/ 恢复成空。"""
    import shutil
    for name in ("key_events.json", "memory_summaries.json", "full_archive.json"):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(os.path.join(DATA, "vectors"), ignore_errors=True)


def _section(title):
    print("\n" + "=" * 56 + f"\n  {title}\n" + "=" * 56)


def run():
    import telegram_bot as t
    from core import memory_vectors, embedder

    # 把样例数据加载进运行时（状态层读全局变量，向量层读 data 文件）
    t.key_events = t.load_key_events()
    t.memory_summaries = t.load_summaries()
    t.full_archive = t.load_archive()

    _section("① 状态层渲染（C1）—— 常驻、只给方向，约定不漏内容")
    print(t.build_stable_memory())

    if not embedder.is_available():
        print("\n[向量不可用] 未装 sentence-transformers/numpy，召回部分跳过。")
        return
    memory_vectors.build_all()

    _section("② A 路自动召回（C2）—— query 字面不重叠也能语义命中")
    for q in ("她家的宠物", "她的工作", "你们的约定"):
        print(f"\n  query: {q}")
        print("  ", t.build_auto_recall(q).replace("\n", "\n  ") or "（无命中）")

    _section("③ B 路记忆检索（C3）+ C6 三态")
    print("\n  search('她养的小动物', detail):")
    print("  ", t.do_search_memory("她养的小动物", "detail"))
    print("\n  search('完全不相干的量子计算xyz', detail) —— 应回“确实没有”:")
    print("  ", t.do_search_memory("完全不相干的量子计算xyz", "detail"))


if __name__ == "__main__":
    _abort_if_real_data()
    _write_sample()
    try:
        run()
    finally:
        _cleanup()
        print("\n[已清理样例数据，data/ 恢复为空]")
