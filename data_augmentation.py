import random
import numpy as np
import re
from data_processor import DataProcessor

class DataAugmenter:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.processor = DataProcessor()
    
    def augment_data(self, prompts, answers, augmentation_factor=1):
        """增强数据"""
        augmented_prompts = []
        augmented_answers = []
        
        for prompt, answer in zip(prompts, answers):
            # 原始数据
            augmented_prompts.append(prompt)
            augmented_answers.append(answer)
            
            # 增强数据
            for _ in range(augmentation_factor):
                problem_type = self.processor._detect_problem_type(prompt)
                aug_prompt, aug_answer = self._augment_by_type(prompt, answer, problem_type)
                if aug_prompt and aug_answer:
                    augmented_prompts.append(aug_prompt)
                    augmented_answers.append(aug_answer)
        
        return augmented_prompts, augmented_answers
    
    def _augment_by_type(self, prompt, answer, problem_type):
        """根据问题类型进行增强"""
        if problem_type == 'bit_manipulation':
            return self._augment_bit_manipulation(prompt, answer)
        elif problem_type == 'encryption':
            return self._augment_encryption(prompt, answer)
        elif problem_type == 'numeral_system':
            return self._augment_numeral_system(prompt, answer)
        elif problem_type == 'unit_conversion':
            return self._augment_unit_conversion(prompt, answer)
        elif problem_type == 'gravity':
            return self._augment_gravity(prompt, answer)
        elif problem_type == 'equation_transformation':
            return self._augment_equation_transformation(prompt, answer)
        else:
            return None, None
    
    def _augment_bit_manipulation(self, prompt, answer):
        """增强位操作问题"""
        # 提取示例
        examples = re.findall(r'(\d{8}) -> (\d{8})', prompt)
        if not examples:
            return None, None
        
        # 生成新的二进制串
        new_input = ''.join([str(random.randint(0, 1)) for _ in range(8)])
        
        # 保持示例不变，替换目标查询
        new_prompt = re.sub(r'Now, determine the output for: (\d{8})', 
                           f'Now, determine the output for: {new_input}', prompt)
        
        # 注意：这里只是示例，实际需要根据规则生成正确答案
        # 由于我们不知道具体规则，这里返回None
        return None, None
    
    def _augment_encryption(self, prompt, answer):
        """增强加密问题"""
        # 提取示例
        examples = re.findall(r'(.*?) -> (.*?)', prompt)
        if not examples:
            return None, None
        
        # 打乱示例顺序
        random.shuffle(examples)
        
        # 重新构建提示
        new_examples = '\n'.join([f'{ex[0]} -> {ex[1]}' for ex in examples])
        new_prompt = re.sub(r'Here are some examples:.*?Now, decrypt the following text:', 
                           f'Here are some examples:\n{new_examples}\n\nNow, decrypt the following text:', 
                           prompt, flags=re.DOTALL)
        
        return new_prompt, answer
    
    def _augment_numeral_system(self, prompt, answer):
        """增强数字系统问题"""
        # 提取示例
        examples = re.findall(r'(\d+) -> (\w+)', prompt)
        if not examples:
            return None, None
        
        # 生成新的数字
        new_number = random.randint(1, 100)
        
        # 替换目标查询
        new_prompt = re.sub(r'Now, write the number (\d+) in the Wonderland numeral system.', 
                           f'Now, write the number {new_number} in the Wonderland numeral system.', prompt)
        
        # 注意：这里只是示例，实际需要根据规则生成正确答案
        # 由于我们不知道具体规则，这里返回None
        return None, None
    
    def _augment_unit_conversion(self, prompt, answer):
        """增强单位转换问题"""
        # 提取示例
        examples = re.findall(r'(\d+\.\d+) m becomes (\d+\.\d+)', prompt)
        if not examples:
            return None, None
        
        # 生成新的测量值
        new_value = round(random.uniform(5, 50), 2)
        
        # 替换目标查询
        new_prompt = re.sub(r'Now, convert the following measurement: (\d+\.\d+) m', 
                           f'Now, convert the following measurement: {new_value} m', prompt)
        
        # 注意：这里只是示例，实际需要根据规则生成正确答案
        # 由于我们不知道具体规则，这里返回None
        return None, None
    
    def _augment_gravity(self, prompt, answer):
        """增强重力问题"""
        # 提取示例
        examples = re.findall(r'For t = (\d+\.\d+)s, distance = (\d+\.\d+) m', prompt)
        if not examples:
            return None, None
        
        # 生成新的时间值
        new_time = round(random.uniform(1, 5), 2)
        
        # 替换目标查询
        new_prompt = re.sub(r'Now, determine the falling distance for t = (\d+\.\d+)s given d = 0\.5\*g\*t\^2.', 
                           f'Now, determine the falling distance for t = {new_time}s given d = 0.5*g*t^2.', prompt)
        
        # 注意：这里只是示例，实际需要根据规则生成正确答案
        # 由于我们不知道具体规则，这里返回None
        return None, None
    
    def _augment_equation_transformation(self, prompt, answer):
        """增强方程转换问题"""
        # 提取示例
        examples = re.findall(r'(.*?) = (.*?)', prompt)
        if not examples:
            return None, None
        
        # 打乱示例顺序
        random.shuffle(examples)
        
        # 重新构建提示
        new_examples = '\n'.join([f'{ex[0]} = {ex[1]}' for ex in examples])
        new_prompt = re.sub(r'Below are a few examples:.*?Now, determine the result for:', 
                           f'Below are a few examples:\n{new_examples}\n\nNow, determine the result for:', 
                           prompt, flags=re.DOTALL)
        
        return new_prompt, answer

class PromptEngineer:
    def __init__(self):
        pass
    
    def create_prompt_template(self, prompt, answer=None, template_type="standard"):
        """创建不同类型的提示模板"""
        if template_type == "standard":
            return self._standard_template(prompt, answer)
        elif template_type == "detailed":
            return self._detailed_template(prompt, answer)
        elif template_type == "concise":
            return self._concise_template(prompt, answer)
        else:
            return self._standard_template(prompt, answer)
    
    def _standard_template(self, prompt, answer):
        """标准模板"""
        template = f"""
        You are given a reasoning problem from Alice's Wonderland. Analyze the examples provided and determine the correct answer for the given query.
        
        Problem:
        {prompt}
        
        {"Answer:" if answer is None else f"Answer: {answer}"}
        """
        return template.strip()
    
    def _detailed_template(self, prompt, answer):
        """详细模板"""
        template = f"""
        Please solve the following reasoning problem from Alice's Wonderland. Follow these steps:
        1. Carefully analyze the examples provided
        2. Identify the underlying pattern or rule
        3. Apply the rule to the given query
        4. Provide your answer
        
        Problem:
        {prompt}
        
        {"Your Answer:" if answer is None else f"Correct Answer: {answer}"}
        """
        return template.strip()
    
    def _concise_template(self, prompt, answer):
        """简洁模板"""
        template = f"""
        Solve this problem:
        {prompt}
        
        {"Answer:" if answer is None else f"Answer: {answer}"}
        """
        return template.strip()
    
    def optimize_prompt(self, prompt, problem_type):
        """根据问题类型优化提示"""
        if problem_type == 'bit_manipulation':
            return self._optimize_bit_manipulation(prompt)
        elif problem_type == 'encryption':
            return self._optimize_encryption(prompt)
        elif problem_type == 'numeral_system':
            return self._optimize_numeral_system(prompt)
        else:
            return prompt
    
    def _optimize_bit_manipulation(self, prompt):
        """优化位操作提示"""
        # 添加位操作相关提示
        optimized_prompt = prompt + "\n\nHint: Consider bitwise operations like AND, OR, XOR, NOT, shifts, and rotations."
        return optimized_prompt
    
    def _optimize_encryption(self, prompt):
        """优化加密提示"""
        # 添加加密相关提示
        optimized_prompt = prompt + "\n\nHint: Look for patterns in the character mappings between encrypted and decrypted text."
        return optimized_prompt
    
    def _optimize_numeral_system(self, prompt):
        """优化数字系统提示"""
        # 添加数字系统相关提示
        optimized_prompt = prompt + "\n\nHint: The Wonderland numeral system appears to be a variation of Roman numerals."
        return optimized_prompt

if __name__ == "__main__":
    # 测试数据增强
    augmenter = DataAugmenter()
    processor = DataProcessor()
    
    # 加载示例数据
    train_df, _ = processor.load_data("nvidia-nemotron-model-reasoning-challenge/train.csv")
    train_prompts = train_df['prompt'].tolist()[:5]
    train_answers = train_df['answer'].tolist()[:5]
    
    # 增强数据
    augmented_prompts, augmented_answers = augmenter.augment_data(train_prompts, train_answers, augmentation_factor=1)
    print(f"原始数据: {len(train_prompts)} samples")
    print(f"增强数据: {len(augmented_prompts)} samples")
    
    # 测试提示工程
    engineer = PromptEngineer()
    example_prompt = train_prompts[0]
    example_answer = train_answers[0]
    
    print("\n标准模板:")
    print(engineer.create_prompt_template(example_prompt, example_answer))
    
    print("\n详细模板:")
    print(engineer.create_prompt_template(example_prompt, example_answer, template_type="detailed"))
    
    print("\n简洁模板:")
    print(engineer.create_prompt_template(example_prompt, example_answer, template_type="concise"))