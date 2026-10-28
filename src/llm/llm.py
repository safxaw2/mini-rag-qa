from openai import OpenAI
from typing import Dict, List,Any, Iterator

class LLM:
    def __init__(self, model:str, api_key:str,base_url:str):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key,base_url=base_url)


    def generate(self, messages:List[Dict], temperature:float=0.5,max_tokens=2000) -> Dict[str,Any]:
        '''
        生成内容
        参数：
        messages:List[Dict] 聊天记录
        temperature:float 采样温度
        max_tokens:int 最大生成长度
        返回：
        Dict[str,Any] 生成的内容
        '''
        response=self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return {
            'content': choice.message.content,
            'finish_reason': choice.finish_reason,
            'tokens_used': {
                'total': response.usage.total_tokens,
                'prompt': response.usage.prompt_tokens,
                'completion': response.usage.completion_tokens
            },
            'model': response.model
        }

    def generate_stream(
        self,
        messages:List[Dict],
        temperature:float=0.5,
        max_tokens:int=2000,
    ) -> Iterator[str]:
        response=self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            content=chunk.choices[0].delta.content
            if content:
                yield content