import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random
import torch

class DataProcessor:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def load_data(self, train_path, test_path=None):
        """加载训练集和测试集"""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path else None
        return train_df, test_df
    
    def preprocess_prompt(self, prompt):
        """预处理提示文本"""
        # 移除多余的空白字符
        prompt = ' '.join(prompt.split())
        # 确保提示以问题结尾
        if not prompt.endswith('?') and not prompt.endswith('.'):
            prompt = prompt + '.'
        return prompt
    
    def create_folds(self, train_df, n_splits=5):
        """创建交叉验证折叠"""
        # 根据问题类型进行分层
        train_df['problem_type'] = train_df['prompt'].apply(self._detect_problem_type)
        
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df['problem_type'])):
            train_df.loc[val_idx, 'fold'] = fold
        
        return train_df
    
    def _detect_problem_type(self, prompt):
        """检测问题类型"""
        prompt_lower = prompt.lower()
        if 'bit manipulation' in prompt_lower:
            return 'bit_manipulation'
        elif 'encryption' in prompt_lower:
            return 'encryption'
        elif 'numeral system' in prompt_lower:
            return 'numeral_system'
        elif 'unit conversion' in prompt_lower:
            return 'unit_conversion'
        elif 'gravitational constant' in prompt_lower:
            return 'gravity'
        elif 'transformation rules' in prompt_lower and 'equation' in prompt_lower:
            return 'equation_transformation'
        else:
            return 'other'
    
    def prepare_training_data(self, train_df, fold=None):
        """准备训练数据"""
        if fold is not None:
            train_data = train_df[train_df['fold'] != fold]
            val_data = train_df[train_df['fold'] == fold]
        else:
            train_data = train_df
            val_data = None
        
        # 预处理训练数据
        train_prompts = [self.preprocess_prompt(p) for p in train_data['prompt'].tolist()]
        train_answers = train_data['answer'].tolist()
        
        if val_data is not None:
            val_prompts = [self.preprocess_prompt(p) for p in val_data['prompt'].tolist()]
            val_answers = val_data['answer'].tolist()
            return train_prompts, train_answers, val_prompts, val_answers
        else:
            return train_prompts, train_answers
    
    def prepare_test_data(self, test_df):
        """准备测试数据"""
        test_prompts = [self.preprocess_prompt(p) for p in test_df['prompt'].tolist()]
        test_ids = test_df['id'].tolist()
        return test_prompts, test_ids
    
    def create_prompt_template(self, prompt, answer=None):
        """创建提示模板"""
        template = f"""
        You are given a reasoning problem from Alice's Wonderland. Analyze the examples provided and determine the correct answer for the given query.
        
        Problem:
        {prompt}
        
        {"Answer:" if answer is None else f"Answer: {answer}"}
        """
        return template.strip()

if __name__ == "__main__":
    # 测试数据处理
    processor = DataProcessor()
    train_df, _ = processor.load_data(
        "nvidia-nemotron-model-reasoning-challenge/train.csv"
    )
    
    # 创建折叠
    train_df = processor.create_folds(train_df)
    
    # 准备训练数据
    train_prompts, train_answers, val_prompts, val_answers = processor.prepare_training_data(train_df, fold=0)
    
    print(f"训练数据: {len(train_prompts)} samples")
    print(f"验证数据: {len(val_prompts)} samples")
    
    # 示例提示
    print("\n示例提示:")
    print(processor.create_prompt_template(train_prompts[0], train_answers[0]))