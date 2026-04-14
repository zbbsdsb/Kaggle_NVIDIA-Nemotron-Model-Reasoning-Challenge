import os
import pandas as pd
import torch
from model_trainer import ModelTrainer
from data_processor import DataProcessor

class InferenceSystem:
    def __init__(self, model_dir="./output"):
        self.model_dir = model_dir
        self.processor = DataProcessor()
        self.trainer = None
    
    def load_models(self, fold=None):
        """加载模型"""
        if fold is not None:
            # 加载指定折叠的模型
            model_path = os.path.join(self.model_dir, f"model_fold_{fold}", "best_model")
            self.trainer = ModelTrainer()
            self.trainer.load_from_checkpoint(model_path)
            return [self.trainer]
        else:
            # 加载所有折叠的模型
            trainers = []
            for fold in range(5):
                model_path = os.path.join(self.model_dir, f"model_fold_{fold}", "best_model")
                if os.path.exists(model_path):
                    trainer = ModelTrainer()
                    trainer.load_from_checkpoint(model_path)
                    trainers.append(trainer)
            return trainers
    
    def predict(self, test_prompts, trainers):
        """生成预测"""
        if len(trainers) == 1:
            # 单个模型预测
            return trainers[0].predict(test_prompts)
        else:
            # 多模型集成预测
            all_preds = []
            for trainer in trainers:
                preds = trainer.predict(test_prompts)
                all_preds.append(preds)
            
            # 多数投票集成
            final_preds = []
            for i in range(len(test_prompts)):
                # 收集所有模型对当前样本的预测
                predictions = [preds[i] for preds in all_preds]
                # 统计每个预测的出现次数
                pred_counts = {}
                for pred in predictions:
                    pred_counts[pred] = pred_counts.get(pred, 0) + 1
                # 选择出现次数最多的预测
                final_pred = max(pred_counts, key=pred_counts.get)
                final_preds.append(final_pred)
            
            return final_preds
    
    def generate_submission(self, test_path, output_path="./submission.csv", fold=None):
        """生成提交文件"""
        # 加载测试数据
        _, test_df = self.processor.load_data(test_path)
        test_prompts, test_ids = self.processor.prepare_test_data(test_df)
        
        # 加载模型
        trainers = self.load_models(fold)
        
        # 生成预测
        print(f"开始推理，使用 {len(trainers)} 个模型")
        predictions = self.predict(test_prompts, trainers)
        
        # 生成提交文件
        submission_df = pd.DataFrame({
            'id': test_ids,
            'answer': predictions
        })
        
        # 保存提交文件
        submission_df.to_csv(output_path, index=False)
        print(f"提交文件已保存至 {output_path}")
        print(f"提交文件形状: {submission_df.shape}")
        print("前5行:")
        print(submission_df.head())
        
        return submission_df
    
    def validate_submission(self, submission_path, test_path):
        """验证提交文件格式"""
        # 加载提交文件
        submission_df = pd.read_csv(submission_path)
        
        # 加载测试文件
        _, test_df = self.processor.load_data(test_path)
        
        # 检查列名
        expected_columns = ['id', 'answer']
        if list(submission_df.columns) != expected_columns:
            print(f"错误: 提交文件列名不正确，期望: {expected_columns}")
            return False
        
        # 检查ID数量
        if len(submission_df) != len(test_df):
            print(f"错误: 提交文件行数不正确，期望: {len(test_df)}，实际: {len(submission_df)}")
            return False
        
        # 检查ID是否匹配
        submission_ids = set(submission_df['id'].tolist())
        test_ids = set(test_df['id'].tolist())
        if submission_ids != test_ids:
            print("错误: 提交文件ID与测试文件ID不匹配")
            return False
        
        # 检查答案列是否为空
        if submission_df['answer'].isnull().any():
            print("错误: 提交文件中存在空答案")
            return False
        
        print("提交文件格式验证通过!")
        return True

if __name__ == "__main__":
    # 测试推理系统
    inference = InferenceSystem(model_dir="./output")
    
    # 生成提交文件
    submission_df = inference.generate_submission(
        test_path="nvidia-nemotron-model-reasoning-challenge/test.csv",
        output_path="./submission.csv"
    )
    
    # 验证提交文件
    inference.validate_submission(
        submission_path="./submission.csv",
        test_path="nvidia-nemotron-model-reasoning-challenge/test.csv"
    )