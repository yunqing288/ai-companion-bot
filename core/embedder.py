"""嵌入模型封装：进程内懒加载单例，对外只暴露 encode 与可用性检测。

红线：本层只做“文本 → 向量”，不做任何关键词/正则/映射的语义推断。
依赖（sentence-transformers / numpy）缺失时按结构容错处理——
报告不可用、encode 返回 None，让上层把“无向量”当成“没搜到”，
而不是退化成关键词匹配重新承担语义理解。
"""
from config import vector_config

_model = None        # SentenceTransformer 单例
_unavailable = False  # 依赖缺失标记，避免反复尝试导入


def _load_model():
    """懒加载模型；依赖缺失则标记不可用并返回 None。"""
    global _model, _unavailable
    if _model is not None:
        return _model
    if _unavailable:
        return None
    try:
        import numpy  # noqa: F401  向量运算依赖
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"[向量] 依赖缺失，向量召回禁用：{e}")
        _unavailable = True
        return None
    _model = SentenceTransformer(vector_config.EMBED_MODEL)
    print(f"[向量] 模型已加载：{vector_config.EMBED_MODEL}")
    return _model


def is_available() -> bool:
    """向量能力是否可用（依赖齐全且模型可加载）。"""
    return _load_model() is not None


def encode(texts):
    """把文本或文本列表编码为归一化向量数组 (n, dim)。

    归一化后点积即 cosine 相似度。不可用时返回 None。
    """
    model = _load_model()
    if model is None:
        return None
    import numpy as np
    batch = [texts] if isinstance(texts, str) else list(texts)
    vecs = model.encode(batch, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")
