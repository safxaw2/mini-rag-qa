import math
import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.openai_embedding import OpenAIEmbedding

try:
    import pytest
except ImportError:  # 未安装 pytest 时仍可直接 python 运行
    pytest = None


# ---------------------------------------------------------------- 配置读取

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_FILE = os.path.join(_REPO_ROOT, ".env")


def _load_env_file(path):
    """极简 .env 解析：支持 KEY=VALUE、注释、export 前缀与引号"""
    values = {}
    if not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip():
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_FILE_ENV = _load_env_file(_ENV_FILE)


def _setting(name, default=None):
    value = os.environ.get(name)
    if value:
        return value
    value = _FILE_ENV.get(name)
    if value:
        return value
    return default


def _env_flag(name, default=True):
    raw = _setting(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


API_KEY = _setting("DASHSCOPE_API_KEY")
BASE_URL = _setting("EMBEDDING_BASE_URL", "https://maas.qianwenaiapi.com/compatible-mode/v1")
MODEL = _setting("EMBEDDING_MODEL", "qwen3.7-text-embedding-flash")
DIMENSION = int(_setting("EMBEDDING_DIMENSION", "1024"))
LIVE_ENABLED = bool(API_KEY) and _env_flag("EMBEDDING_LIVE", default=True)

LIVE_SKIP_REASON = (
    "未配置真实接口凭据：请在 .env 或环境变量中设置 DASHSCOPE_API_KEY"
    if not API_KEY
    else "EMBEDDING_LIVE=0，已跳过真实接口调用"
)


def live_suite(cls):
    """真实接口用例：pytest 下 skipif，裸跑时由手写 runner 读取 _skip_reason"""
    reason = None if LIVE_ENABLED else LIVE_SKIP_REASON
    if pytest is not None:
        cls = pytest.mark.skipif(not LIVE_ENABLED, reason=LIVE_SKIP_REASON)(cls)
    cls._skip_reason = reason
    return cls


_LIVE_CACHE = {}


def _live_embedding(dimension=None):
    """惰性构造真实实例，避免无 key 时在导入阶段报错"""
    if dimension not in _LIVE_CACHE:
        _LIVE_CACHE[dimension] = OpenAIEmbedding(
            api_key=API_KEY,
            model=MODEL,
            base_url=BASE_URL,
            dimension=dimension,
        )
    return _LIVE_CACHE[dimension]


# ---------------------------------------------------------------- 测试辅助

def _cosine(a, b):
    """纯标准库余弦相似度"""
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


# ---------------------------------------------------------------- 测试语料

# 由测试自己准备：三个语义组，每组三条，用于验证分组相似度
TEXTS = {
    "cat": "猫是一种常见的宠物，喜欢晒太阳和睡觉。",
    "dog": "狗是人类忠诚的朋友，喜欢和主人一起玩耍。",
    "rabbit": "兔子爱吃胡萝卜，耳朵又长又软。",
    "car": "汽车在高速公路上行驶，速度很快。",
    "train": "高铁列车准时到达车站，乘客有序下车。",
    "plane": "飞机起飞后穿过云层，飞向远方。",
    "ai": "人工智能正在改变我们的生活方式和工作方式。",
    "quantum": "量子计算机利用量子比特进行并行计算。",
    "robot": "机器人可以协助人类完成重复性的工作。",
}

GROUPS = {
    "动物": ["cat", "dog", "rabbit"],
    "交通工具": ["car", "train", "plane"],
    "科技": ["ai", "quantum", "robot"],
}


# ---------------------------------------------------------------- 真实接口用例

@live_suite
class TestEmbedText:
    """调用 embed_text / embed 对自备文本做向量化"""

    def test_embed_text_returns_vector(self):
        text = "猫是一种常见的宠物，喜欢晒太阳和睡觉。"
        vec = _live_embedding().embed_text(text)

        _assert_vector(vec, DIMENSION)
        print(f"\n[embed_text] 维度={len(vec)}，前 5 维={[round(x, 6) for x in vec[:5]]}")
        print(f"[embed_text] 范数={math.sqrt(sum(x * x for x in vec)):.6f}")

    def test_embed_matches_embed_text(self):
        text = "同一段文本用两个入口应该得到一致的向量。"
        emb = _live_embedding()
        via_embed = emb.embed(text)
        via_embed_text = emb.embed_text(text)

        similarity = _cosine(via_embed, via_embed_text)
        print(f"\n[embed vs embed_text] 余弦相似度={similarity:.8f}")
        assert similarity > 0.9999


@live_suite
class TestEmbedBatch:
    """调用 embed_batch 对自备文本批量向量化"""

    def test_batch_matches_single(self):
        keys = ["cat", "dog", "car"]
        texts = [TEXTS[k] for k in keys]
        emb = _live_embedding()

        batch = emb.embed_batch(texts)
        singles = [emb.embed_text(t) for t in texts]

        assert len(batch) == len(texts)
        for i, (key, row, single) in enumerate(zip(keys, batch, singles)):
            _assert_vector(row, DIMENSION)
            similarity = _cosine(row, single)
            print(f"\n[批量 vs 单条] {key}: 余弦={similarity:.8f}")
            # 实测批量与单条并非逐位一致（约 0.9986），用余弦判断语义等价即可
            assert similarity > 0.99, f"{key} 的批量向量与单条向量差异过大"

    def test_semantic_groups(self):
        texts, labels = [], []
        for group, keys in GROUPS.items():
            for key in keys:
                texts.append(TEXTS[key])
                labels.append(group)

        rows = _live_embedding().embed_batch(texts)
        assert len(rows) == len(texts)
        for row in rows:
            _assert_vector(row, DIMENSION)

        within, between = [], []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                similarity = _cosine(rows[i], rows[j])
                (within if labels[i] == labels[j] else between).append(similarity)

        mean_within = sum(within) / len(within)
        mean_between = sum(between) / len(between)
        print(f"\n[语义分组] 组内均值={mean_within:.4f}，组间均值={mean_between:.4f}")

        assert mean_within > mean_between + 0.05, (
            f"组内相似度应明显高于组间：{mean_within:.4f} vs {mean_between:.4f}"
        )

    def test_chinese_english_semantic_match(self):
        texts = ["今天天气很好，适合出门散步。", "The weather is nice today, perfect for a walk."]
        rows = _live_embedding().embed_batch(texts)
        similarity = _cosine(rows[0], rows[1])
        print(f"\n[中英同义] 余弦={similarity:.4f}")
        assert similarity > 0.55

    def test_custom_dimension_512(self):
        vec = _live_embedding(dimension=512).embed_text("自定义向量维度测试。")
        _assert_vector(vec, 512)
        print(f"\n[512 维] 实际维度={len(vec)}")

    def test_batch_limit_20_rows(self):
        texts = [
            f"这是第 {i + 1} 条用于测试批量上限的文本，内容各不相同。"
            for i in range(20)
        ]
        rows = _live_embedding().embed_batch(texts)
        assert len(rows) == 20
        for row in rows:
            _assert_vector(row, DIMENSION)
        print(f"\n[20 条批量] 返回 {len(rows)} 行，每行 {len(rows[0])} 维")


if __name__ == "__main__":
    # 无 pytest 时也可直接运行：手动收集并执行所有 test_ 方法
    failures = []
    total = 0
    skipped = 0
    for name, obj in list(globals().items()):
        if not (isinstance(obj, type) and name.startswith("Test")):
            continue
        class_skip = getattr(obj, "_skip_reason", None)
        for method_name in dir(obj):
            if not method_name.startswith("test_"):
                continue
            method = getattr(obj, method_name)
            if not isinstance(method, types.FunctionType):
                continue
            total += 1
            if class_skip:
                skipped += 1
                print(f"SKIP  {name}.{method_name}: {class_skip}")
                continue
            try:
                method(obj())
                print(f"PASS  {name}.{method_name}")
            except AssertionError as exc:
                failures.append((name, method_name, exc))
                print(f"FAIL  {name}.{method_name}: {exc}")
            except Exception as exc:  # 网络/鉴权等错误
                failures.append((name, method_name, exc))
                print(f"ERROR {name}.{method_name}: {exc!r}")

    passed = total - skipped - len(failures)
    print(f"\n{passed}/{total - skipped} 通过，{skipped} 跳过")
    if failures:
        raise SystemExit(1)
