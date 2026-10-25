"""LocalEmbedding 测试。

测试语料全部在文件内自行生成，不依赖外部数据；模型使用
BAAI/bge-small-zh-v1.5（中文小模型），并统计模型加载、单条调用、
批量调用各自花费的时间。

运行方式：
    pytest tests/test_local_embedding.py -s
    # 或未安装 pytest 时直接运行
    python tests/test_local_embedding.py

可通过环境变量 LOCAL_EMBEDDING_DEVICE 指定设备（默认自动选择）。
"""

import math
import os
import sys
import time
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.local_embedding import LocalEmbedding

try:
    import pytest
except ImportError:  # 未安装 pytest 时仍可直接 python 运行
    pytest = None


# ---------------------------------------------------------------- 测试配置

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EXPECTED_DIM = 512


def _pick_device() -> str:
    """优先读取环境变量；未指定时按 CUDA 是否可用自动选择。"""
    configured = os.environ.get("LOCAL_EMBEDDING_DEVICE")
    if configured:
        return configured
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _pick_device()


# ---------------------------------------------------------------- 测试语料

# 三个语义分组，每组三条，用于验证组内相似度高于组间相似度
TEXTS = {
    # 动物
    "cat": "猫是一种常见的宠物，喜欢晒太阳和睡觉。",
    "dog": "狗是人类忠诚的朋友，喜欢和主人一起玩耍。",
    "rabbit": "兔子爱吃胡萝卜，耳朵又长又软。",
    # 交通工具
    "car": "汽车在高速公路上行驶，速度很快。",
    "train": "高铁列车准时到达车站，乘客有序下车。",
    "plane": "飞机起飞后穿过云层，飞向远方。",
    # 科技
    "ai": "人工智能正在改变我们的生活方式和工作方式。",
    "quantum": "量子计算机利用量子比特进行并行计算。",
    "robot": "机器人可以协助人类完成重复性的工作。",
}

GROUPS = {
    "动物": ["cat", "dog", "rabbit"],
    "交通工具": ["car", "train", "plane"],
    "科技": ["ai", "quantum", "robot"],
}


def _make_batch_texts(count: int = 32):
    """生成用于批量耗时测试的中文文本。"""
    return [
        f"这是第 {i + 1} 条测试文本，主题编号为 {i % 5}，"
        f"用于衡量本地嵌入模型处理批量文本的耗时。"
        for i in range(count)
    ]


# ---------------------------------------------------------------- 测试辅助

_CACHE = {}
_TIMING = {}


def _get_embedding() -> LocalEmbedding:
    """惰性加载模型并缓存，避免每个用例重复加载；同时记录加载耗时。"""
    if "embedding" not in _CACHE:
        started = time.perf_counter()
        _CACHE["embedding"] = LocalEmbedding(model_name=MODEL_NAME, device=DEVICE)
        _TIMING["load"] = time.perf_counter() - started
        print(f"\n[加载] {MODEL_NAME} 设备={DEVICE} 耗时={_TIMING['load']:.3f}s")
    return _CACHE["embedding"]


def _timed(fn):
    """执行一次调用并返回 (结果, 耗时秒)。"""
    started = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - started


def _cosine(a, b):
    """纯标准库余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _assert_vector(vec, dimension):
    assert isinstance(vec, list), f"向量应为 list，实际 {type(vec)!r}"
    assert len(vec) == dimension, f"维度应为 {dimension}，实际 {len(vec)}"
    assert all(isinstance(x, (int, float)) for x in vec), "向量元素应为数字"
    assert all(math.isfinite(float(x)) for x in vec), "向量包含 NaN/Inf"
    assert math.sqrt(sum(float(x) ** 2 for x in vec)) > 1e-6, "向量范数为 0"


# ---------------------------------------------------------------- 基础功能用例

class TestLocalEmbeddingBasics:
    """验证模型信息、单条与批量嵌入的正确性。"""

    def test_model_info(self):
        emb = _get_embedding()

        dimension = emb.get_dimension()
        name = emb.get_name()
        print(f"\n[模型信息] 名称={name} 维度={dimension}")
        assert dimension == EXPECTED_DIM, f"维度应为 {EXPECTED_DIM}，实际 {dimension}"
        assert name == MODEL_NAME

    def test_embed_text_returns_vector(self):
        emb = _get_embedding()
        text = TEXTS["cat"]

        vec, elapsed = _timed(lambda: emb.embed_text(text))
        _assert_vector(vec, EXPECTED_DIM)
        print(f"\n[embed_text] 文本={text!r} 维度={len(vec)} 耗时={elapsed * 1000:.2f}ms")
        print(f"[embed_text] 前 5 维={[round(x, 6) for x in vec[:5]]}")

    def test_embed_matches_embed_text(self):
        emb = _get_embedding()
        text = "同一段文本用两个入口应该得到一致的向量。"

        via_embed, t1 = _timed(lambda: emb.embed(text))
        via_embed_text, t2 = _timed(lambda: emb.embed_text(text))
        similarity = _cosine(via_embed, via_embed_text)

        print(f"\n[embed vs embed_text] 余弦={similarity:.8f} "
              f"耗时={t1 * 1000:.2f}ms / {t2 * 1000:.2f}ms")
        assert similarity > 0.9999

    def test_embed_batch_returns_rows(self):
        emb = _get_embedding()
        texts = _make_batch_texts(4)

        rows, elapsed = _timed(lambda: emb.embed_batch(texts))
        assert len(rows) == len(texts)
        for row in rows:
            _assert_vector(row, EXPECTED_DIM)
        print(f"\n[embed_batch] {len(texts)} 条文本，耗时={elapsed * 1000:.2f}ms，"
              f"每条均摊={elapsed / len(texts) * 1000:.2f}ms")

    def test_batch_matches_single(self):
        emb = _get_embedding()
        keys = ["cat", "dog", "car", "ai"]
        texts = [TEXTS[k] for k in keys]

        batch = emb.embed_batch(texts)
        singles = [emb.embed_text(text) for text in texts]

        assert len(batch) == len(texts)
        for key, row, single in zip(keys, batch, singles):
            _assert_vector(row, EXPECTED_DIM)
            similarity = _cosine(row, single)
            print(f"[批量 vs 单条] {key}: 余弦={similarity:.8f}")
            assert similarity > 0.99, f"{key} 的批量向量与单条向量差异过大"

    def test_semantic_groups(self):
        emb = _get_embedding()
        texts, labels = [], []
        for group, keys in GROUPS.items():
            for key in keys:
                texts.append(TEXTS[key])
                labels.append(group)

        rows, elapsed = _timed(lambda: emb.embed_batch(texts))
        assert len(rows) == len(texts)

        within, between = [], []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                similarity = _cosine(rows[i], rows[j])
                (within if labels[i] == labels[j] else between).append(similarity)

        mean_within = sum(within) / len(within)
        mean_between = sum(between) / len(between)
        print(f"\n[语义分组] 组内均值={mean_within:.4f}，组间均值={mean_between:.4f}，"
              f"耗时={elapsed * 1000:.2f}ms")
        assert mean_within > mean_between + 0.05, (
            f"组内相似度应明显高于组间：{mean_within:.4f} vs {mean_between:.4f}"
        )

    def test_similar_vs_dissimilar(self):
        emb = _get_embedding()
        similar_a = "今天天气很好，适合出门散步。"
        similar_b = "今天阳光明媚，非常适合到户外走走。"
        dissimilar = "股票市场今天大幅下跌，投资者情绪谨慎。"

        rows = emb.embed_batch([similar_a, similar_b, dissimilar])
        sim = _cosine(rows[0], rows[1])
        diff = _cosine(rows[0], rows[2])
        print(f"\n[相似度] 近义={sim:.4f}，无关={diff:.4f}")
        assert sim > diff + 0.05, f"近义句相似度应高于无关句：{sim:.4f} vs {diff:.4f}"


# ---------------------------------------------------------------- 耗时用例

class TestPerformance:
    """统计模型调用耗时（结果打印，不做严格的性能阈值断言）。"""

    def test_model_load_time(self):
        _get_embedding()
        load = _TIMING["load"]
        print(f"\n[耗时] 模型加载：{load:.3f}s（设备={DEVICE}）")
        assert load > 0

    def test_single_call_latency(self):
        emb = _get_embedding()
        text = TEXTS["quantum"]

        emb.embed_text(text)  # 预热，排除首次调用的额外开销
        times = []
        for _ in range(5):
            _, elapsed = _timed(lambda: emb.embed_text(text))
            times.append(elapsed)

        avg = sum(times) / len(times)
        print(f"\n[耗时] 单条调用 5 次：平均={avg * 1000:.2f}ms，"
              f"最快={min(times) * 1000:.2f}ms，最慢={max(times) * 1000:.2f}ms")
        assert all(t > 0 for t in times)

    def test_batch_throughput(self):
        emb = _get_embedding()
        texts = _make_batch_texts(32)

        _, batch_elapsed = _timed(lambda: emb.embed_batch(texts))
        single_times = []
        for text in texts:
            _, elapsed = _timed(lambda t=text: emb.embed_text(t))
            single_times.append(elapsed)
        single_total = sum(single_times)

        print(f"\n[耗时] 批量 {len(texts)} 条：{batch_elapsed * 1000:.2f}ms "
              f"（每条均摊 {batch_elapsed / len(texts) * 1000:.2f}ms）")
        print(f"[耗时] 逐条调用 {len(texts)} 次：{single_total * 1000:.2f}ms "
              f"（每条均摊 {single_total / len(texts) * 1000:.2f}ms）")
        print(f"[耗时] 批量加速比：{single_total / batch_elapsed:.2f}x")
        assert batch_elapsed > 0 and single_total > 0

    def test_timing_summary(self):
        _get_embedding()
        emb = _get_embedding()
        _, single_elapsed = _timed(lambda: emb.embed_text(TEXTS["ai"]))
        _, batch_elapsed = _timed(lambda: emb.embed_batch(_make_batch_texts(32)))

        print("\n========== 耗时汇总 ==========")
        print(f"模型          : {MODEL_NAME}")
        print(f"设备          : {DEVICE}")
        print(f"模型加载      : {_TIMING['load']:.3f}s")
        print(f"单条嵌入      : {single_elapsed * 1000:.2f}ms")
        print(f"32 条批量嵌入 : {batch_elapsed * 1000:.2f}ms")
        print("==============================")
        assert _TIMING["load"] > 0


if __name__ == "__main__":
    # 无 pytest 时也可直接运行：手动收集并执行所有 test_ 方法
    failures = []
    total = 0
    for name, obj in list(globals().items()):
        if not (isinstance(obj, type) and name.startswith("Test")):
            continue
        for method_name in dir(obj):
            if not method_name.startswith("test_"):
                continue
            method = getattr(obj, method_name)
            if not isinstance(method, types.FunctionType):
                continue
            total += 1
            try:
                method(obj())
                print(f"PASS  {name}.{method_name}")
            except AssertionError as exc:
                failures.append((name, method_name, exc))
                print(f"FAIL  {name}.{method_name}: {exc}")
            except Exception as exc:  # 模型下载/设备等错误
                failures.append((name, method_name, exc))
                print(f"ERROR {name}.{method_name}: {exc!r}")

    passed = total - len(failures)
    print(f"\n{passed}/{total} 通过")
    if failures:
        raise SystemExit(1)
