import os
import json
import numpy as np
import pandas as pd
from data_processor import DataProcessor

class CVSystem:
    def __init__(self, n_folds=5, output_dir="./output"):
        self.n_folds = n_folds
        self.output_dir = output_dir
        self.processor = DataProcessor()
        self.oof_predictions = []
        self.oof_scores = []
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def run_cv(self, train_path, model_trainer):
        """运行交叉验证"""
        # 加载数据
        train_df, _ = self.processor.load_data(train_path)
        train_df = self.processor.create_folds(train_df, n_splits=self.n_folds)
        
        # 对每个折叠进行训练和验证
        for fold in range(self.n_folds):
            print(f"\n=== 开始训练 Fold {fold} ===")
            
            # 准备数据
            train_prompts, train_answers, val_prompts, val_answers = self.processor.prepare_training_data(train_df, fold=fold)
            
            # 训练模型
            model_path = os.path.join(self.output_dir, f"model_fold_{fold}")
            os.makedirs(model_path, exist_ok=True)
            
            # 训练并获取验证预测
            val_preds = model_trainer.train(
                train_prompts, train_answers,
                val_prompts, val_answers,
                output_dir=model_path,
                fold=fold
            )
            
            # 计算分数
            score = self._calculate_score(val_answers, val_preds)
            self.oof_scores.append(score)
            print(f"Fold {fold} 分数: {score:.4f}")
            
            # 保存OOF预测
            val_ids = train_df[train_df['fold'] == fold]['id'].tolist()
            self._save_oof_predictions(val_ids, val_preds, val_answers, fold)
        
        # 计算总体OOF分数
        self._calculate_overall_oof_score()
        return self.oof_predictions, self.oof_scores
    
    def _calculate_score(self, true_answers, predictions):
        """计算准确率分数"""
        correct = 0
        for true, pred in zip(true_answers, predictions):
            if str(true).strip() == str(pred).strip():
                correct += 1
        return correct / len(true_answers)
    
    def _save_oof_predictions(self, ids, predictions, true_answers, fold):
        """保存OOF预测"""
        oof_df = pd.DataFrame({
            'id': ids,
            'true_answer': true_answers,
            'prediction': predictions,
            'fold': [fold] * len(ids)
        })
        
        oof_df.to_csv(
            os.path.join(self.output_dir, f"oof_fold_{fold}.csv"),
            index=False
        )
        
        # 添加到总体OOF预测
        self.oof_predictions.extend(list(zip(ids, predictions, true_answers)))
    
    def _calculate_overall_oof_score(self):
        """计算总体OOF分数"""
        if not self.oof_predictions:
            return 0.0
        
        correct = 0
        for _, pred, true in self.oof_predictions:
            if str(true).strip() == str(pred).strip():
                correct += 1
        
        overall_score = correct / len(self.oof_predictions)
        print(f"\n=== 总体OOF分数: {overall_score:.4f} ===")
        print(f"各折叠分数: {[f'{s:.4f}' for s in self.oof_scores]}")
        print(f"分数标准差: {np.std(self.oof_scores):.4f}")
        
        # 保存OOF结果
        oof_df = pd.DataFrame(
            self.oof_predictions,
            columns=['id', 'prediction', 'true_answer']
        )
        oof_df.to_csv(
            os.path.join(self.output_dir, "oof_predictions.csv"),
            index=False
        )
        
        # 保存分数
        scores = {
            'overall_score': overall_score,
            'fold_scores': self.oof_scores,
            'std': float(np.std(self.oof_scores))
        }
        with open(os.path.join(self.output_dir, "oof_scores.json"), 'w') as f:
            json.dump(scores, f, indent=2)
        
        return overall_score
    
    def load_oof_predictions(self):
        """加载OOF预测"""
        oof_path = os.path.join(self.output_dir, "oof_predictions.csv")
        if os.path.exists(oof_path):
            return pd.read_csv(oof_path)
        return None

if __name__ == "__main__":
    # 示例用法
    class DummyModelTrainer:
        def train(self, train_prompts, train_answers, val_prompts, val_answers, **kwargs):
            # 模拟预测
            return [random.choice(val_answers) for _ in val_prompts]
    
    import random
    random.seed(42)
    
    cv = CVSystem()
    cv.run_cv(
        "nvidia-nemotron-model-reasoning-challenge/train.csv",
        DummyModelTrainer()
    )