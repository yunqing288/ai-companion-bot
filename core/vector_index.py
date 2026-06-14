"""通用向量集合：一个集合 = 一类记忆的向量库。

职责单一：持久化（id + 文本哈希 + 向量）、按 id 增量同步、cosine top-k 搜索。
不含任何业务语义；放什么进来、怎么解释结果由业务侧（memory_vectors）决定。

落盘结构（每个集合一个子目录）：
  data/vectors/<name>/meta.json    ids 顺序、id→原文、id→文本哈希
  data/vectors/<name>/vectors.npy  与 ids 行对齐的归一化向量矩阵
"""
import os
import json
import hashlib

from core import embedder


def _hash(text: str) -> str:
    """文本指纹：内容不变则哈希不变，用于跳过重复编码。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class VectorIndex:
    def __init__(self, name: str, root_dir: str):
        self.name = name
        self.dir = os.path.join(root_dir, name)
        self.ids = []        # list[str]，与向量矩阵行一一对齐
        self.texts = {}      # id -> 原文（用于返回结果）
        self.hashes = {}     # id -> 文本哈希（判断是否需重新编码）
        self.vectors = None  # numpy (n, dim) 或 None
        self._load()

    # ── 持久化 ────────────────────────────────────────────
    def _meta_path(self):
        return os.path.join(self.dir, "meta.json")

    def _vec_path(self):
        return os.path.join(self.dir, "vectors.npy")

    def _load(self):
        """从磁盘载入元数据与向量；不存在则保持空。"""
        try:
            with open(self._meta_path(), encoding="utf-8") as f:
                meta = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        self.ids = meta.get("ids", [])
        self.texts = meta.get("texts", {})
        self.hashes = meta.get("hashes", {})
        self._load_vectors()

    def _load_vectors(self):
        import numpy as np
        try:
            self.vectors = np.load(self._vec_path())
        except (FileNotFoundError, OSError, ValueError):
            self.vectors = None

    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        meta = {"ids": self.ids, "texts": self.texts, "hashes": self.hashes}
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        if self.vectors is not None:
            import numpy as np
            np.save(self._vec_path(), self.vectors)

    # ── 增量同步 ──────────────────────────────────────────
    def _encode_changed(self, to_encode):
        """只编码内容有变化的条目，返回 id -> 向量。"""
        if not to_encode:
            return {}
        vecs = embedder.encode([t for _, t in to_encode])
        if vecs is None:
            return {}
        return {eid: vecs[i] for i, (eid, _) in enumerate(to_encode)}

    def sync(self, items):
        """与给定全集 [(id, text)] 对齐：新增/变更重新编码，消失的删除。

        增量优化：未变更的条目复用旧向量，不重复 encode。
        """
        if not embedder.is_available():
            return False
        import numpy as np
        old_pos = {eid: i for i, eid in enumerate(self.ids)}
        changed = [(eid, t) for eid, t in items if self.hashes.get(eid) != _hash(t)]
        fresh = self._encode_changed(changed)
        rows, ids, texts, hashes = [], [], {}, {}
        for eid, text in items:
            vec = self._pick_vector(eid, fresh, old_pos)
            if vec is None:
                continue
            rows.append(vec)
            ids.append(eid)
            texts[eid] = text
            hashes[eid] = _hash(text)
        self.ids, self.texts, self.hashes = ids, texts, hashes
        self.vectors = np.vstack(rows) if rows else None
        self.save()
        return True

    def _pick_vector(self, eid, fresh, old_pos):
        """新编码优先，否则复用旧矩阵中的行；都没有则返回 None。"""
        if eid in fresh:
            return fresh[eid]
        if self.vectors is not None and eid in old_pos:
            return self.vectors[old_pos[eid]]
        return None

    # ── 检索 ──────────────────────────────────────────────
    def search(self, query: str, top_k: int, floor: float = 0.0):
        """返回 [{id, text, score}]，按相似度降序，过滤掉低于 floor 的。"""
        if self.vectors is None or not self.ids:
            return []
        qv = embedder.encode(query)
        if qv is None:
            return []
        import numpy as np
        scores = self.vectors @ qv[0]  # 向量已归一化，点积即 cosine
        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in order:
            score = float(scores[i])
            if score < floor:
                break
            eid = self.ids[i]
            results.append({"id": eid, "text": self.texts.get(eid, ""), "score": score})
        return results
