'''
检索向量数据库
'''
from typing import List,Dict,Any,Iterator
from src.llm.llm import LLM

def format_context(results:List[Dict[str,Any]],max_chars:int=6000) -> str:
    '''
    格式化上下文
    '''
    seen=set()
    parts=[]
    total=0

    for result in results:
        content=result.get('content',' ').strip()
        if not content or content in seen:
            continue
        next_total=total+len(content)+(2 if len(parts) else 0)
        if next_total>max_chars:
            continue
        seen.add(content)
        parts.append(content)
        total=next_total

    return '\n\n'.join(parts)

def build_prompt(
        question:str,
        context:str
    )->str:
    '''
    构建答案提示词
    '''
    instruction=(
        '你是一个基于文档的问答助手。请先仔细查找资料中是否包含问题的答案。'
        '如果资料中明确包含答案，必须直接回答该答案，不要因为资料简短或缺少额外背景而回答“不知道”。'
        '只有当资料确实没有提及、也无法根据资料确定答案时，才回答“不知道”。'
        '回答应简洁，并且只能依据提供的资料。'
    )
    return (
        f'{instruction}\n\n'
        f'--- 资料开始 ---\n{context}\n--- 资料结束 ---\n\n'
        f'问题：{question}\n'
        '请直接给出答案：'
    )


def generate_answer(
        llm:LLM,
        question:str,
        result:List[Dict[str,Any]],
        max_token:int=2000,
        temperature:float=0.7
    ) -> Dict[str,Any]:
    '''
    生成回答
    ARGS:
        llm: LLM
        question: 问题
        result: 检索结果
        max_tokens: 最大生成长度
        temperature: 采样温度
    RETURNS:
        Dict[str,Any]
    '''
    if not result:
        return {
            'question':question,
            'answer':'没有找到答案',
            'context':"",
        }

    #格式化上下文
    context=format_context(result)

    #构建提示词
    prompt=build_prompt(question,context)
    #调用llm
    message=[
        {"role":"user","content":prompt}
    ]
    result=llm.generate(message,temperature=temperature,max_tokens=max_token)
    #返回结果
    return {
        'answer':result['content'],
        'question':question,
        'context':context,
        'tokens_used':result['tokens_used'],
        'model':result['model']
    }


def generate_answer_stream(
        llm:LLM,
        question:str,
        result:List[Dict[str,Any]],
        max_token:int=2000,
        temperature:float=0.7
    )->Iterator[str]:
    if not result:
        yield '没有找到答案'
        return

    context=format_context(result)
    prompt=build_prompt(question,context)
    yield from llm.generate_stream(
        [{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_token,
    )