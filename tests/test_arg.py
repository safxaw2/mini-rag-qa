from src.llm.rag import build_prompt, format_context, generate_answer


def test_format_context_strips_empty_and_duplicate_content():
    results = [
        {"content": "  第一段  "},
        {"content": ""},
        {"content": "第一段"},
        {"content": "第二段"},
    ]

    assert format_context(results) == "第一段\n\n第二段"


def test_format_context_respects_max_chars():
    results = [
        {"content": "第一段"},
        {"content": "超出长度的内容"},
        {"content": "末段"},
    ]

    assert format_context(results, max_chars=10) == "第一段\n\n末段"


def test_format_context_keeps_later_relevant_chunk_with_default_limit():
    results = [
        {"content": "无关内容" * 242},
        {"content": "系统要求：内存至少 8 GB。" + "其他内容" * 150},
        {"content": "文档结束"},
    ]

    context = format_context(results)

    assert "系统要求：内存至少 8 GB。" in context
    assert "文档结束" in context


def test_build_prompt_includes_context_and_question():
    prompt = build_prompt("问题是什么？", "相关文档内容")

    assert isinstance(prompt, str)
    assert "相关文档内容" in prompt
    assert "问题是什么？" in prompt
    assert "不知道" in prompt
    assert "必须直接回答该答案" in prompt
    assert "不要因为资料简短" in prompt


def test_generate_answer_returns_not_found_without_calling_llm():
    class UnusedLLM:
        def generate(self, *args, **kwargs):
            raise AssertionError("LLM should not be called without results")

    assert generate_answer(UnusedLLM(), "问题", []) == {
        "question": "问题",
        "answer": "没有找到答案",
        "context": "",
    }


def test_generate_answer_passes_prompt_options_and_returns_llm_result():
    class StubLLM:
        def __init__(self):
            self.call = None

        def generate(self, messages, temperature, max_tokens):
            self.call = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            return {
                "content": "生成的回答",
                "tokens_used": {"total": 12},
                "model": "test-model",
            }

    llm = StubLLM()
    answer = generate_answer(
        llm,
        "问题",
        [{"content": "参考内容"}],
        max_token=128,
        temperature=0.2,
    )

    assert answer == {
        "answer": "生成的回答",
        "question": "问题",
        "context": "参考内容",
        "tokens_used": {"total": 12},
        "model": "test-model",
    }
    assert llm.call["messages"][0]["role"] == "user"
    assert "参考内容" in llm.call["messages"][0]["content"]
    assert "问题" in llm.call["messages"][0]["content"]
    assert "必须直接回答该答案" in llm.call["messages"][0]["content"]
    assert llm.call["temperature"] == 0.2
    assert llm.call["max_tokens"] == 128
