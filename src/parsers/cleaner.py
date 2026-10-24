import re
from typing import List

class TextCleaner:
    '''
    文本清洗器  
    '''
    def __init__(self,custom_patterns:List[str]=None):
        '''
        Args:
            custom_patterns (List[str], optional): 自定义正则表达式列表. Defaults to None.
        '''
        #常见的无意义模式
        self.patterns_to_remove = [
            r'^\s*\d+\s*$',        # 单独的页码
            r'^\s*[-=_]+\s*$',      # 分隔线
            r'^\s*第\s*\d+\s*页\s*$',   # 中文页码标记
            r'^\s*Page\s*\d+\s*$',    # 英文页码
            r'^\s*\d+\s*/\s*\d+\s*$',   # 页码格式：1/10
        ]
        #t添加自定义正则表达式
        if custom_patterns:
            self.patterns_to_remove.extend(custom_patterns)

    def clean(self,text:str) -> str:
        '''
        清洗文本
        Args:
            text (str): 待清洗的文本
        Returns:
            str: 清洗后的文本
        '''
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            #清除空白行
            line = line.strip()
            if not line:
                continue
            #清除自定义正则表达式
            for pattern in self.patterns_to_remove:
                if re.match(pattern,line): 
                    break
            else: #没有匹配到自定义正则表达式
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def normalize_withespace(self,text:str) -> str:
        '''
        去除多余的空白
        - 将多个连续的空白字符替换为一个空格
        - 将行首和行尾的空白字符去除
        - 将多个换行符替换为一个换行符
        Args:
            text (str): 待处理的文本
        Returns:
            str: 处理后的文本
        '''
        #去除多余的空白
        text = re.sub(r'\n{3,}', '\n\n', text)      # 多个空行压成一个
        text = re.sub(r'[ \t]+', ' ', text)          # 连续水平空白压成一个
        text = re.sub(r'[ \t]+\n', '\n', text)
        #去除行首和行尾的空白字符
        return text.strip()

    def remove_extra_spaces(self,text:str) -> str:
        '''
        去除多余的空白字符
        Args:
            text (str): 待处理的文本
        Returns:
            str: 处理后的文本
        '''
        #去除多余的字符
        text = re.sub(r'[ \t]+', ' ', text)
        return text

    def full_clean(self,text:str) -> str:
        '''
        文本清理
        Args:
            text (str): 待清理的文本
        Returns:
            str: 清理后的文本
        '''
        text=self.clean(text)
        text=self.normalize_withespace(text)
        text=self.remove_extra_spaces(text)
        return text

        
