import os
import argparse
from cv_system import CVSystem
from model_trainer import ModelTrainer
from inference import InferenceSystem

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Nemotron推理挑战流水线")
    parser.add_argument('--train_path', type=str, default="nvidia-nemotron-model-reasoning-challenge/train.csv", help="训练数据路径")
    parser.add_argument('--test_path', type=str, default="nvidia-nemotron-model-reasoning-challenge/test.csv", help="测试数据路径")
    parser.add_argument('--output_dir', type=str, default="./output", help="输出目录")
    parser.add_argument('--n_folds', type=int, default=5, help="交叉验证折叠数")
    parser.add_argument('--lora_rank', type=int, default=32, help="LoRA rank")
    parser.add_argument('--batch_size', type=int, default=1, help="批量大小")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="梯度累积步数")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="学习率")
    parser.add_argument('--epochs', type=int, default=2, help="训练轮次")
    parser.add_argument('--run_cv', action='store_true', help="运行交叉验证")
    parser.add_argument('--run_inference', action='store_true', help="运行推理")
    parser.add_argument('--submission_path', type=str, default="./submission.csv", help="提交文件路径")
    return parser.parse_args()

def run_cv(args):
    """运行交叉验证"""
    print("=== 开始运行交叉验证 ===")
    
    # 创建CV系统
    cv = CVSystem(n_folds=args.n_folds, output_dir=args.output_dir)
    
    # 创建模型训练器
    trainer = ModelTrainer(lora_rank=args.lora_rank)
    
    # 定义训练函数
    def train_fn(train_prompts, train_answers, val_prompts, val_answers, output_dir, fold):
        return trainer.train(
            train_prompts, train_answers,
            val_prompts, val_answers,
            output_dir=output_dir,
            fold=fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate
        )
    
    # 运行交叉验证
    cv.run_cv(args.train_path, train_fn)
    print("=== 交叉验证完成 ===")

def run_inference(args):
    """运行推理"""
    print("=== 开始运行推理 ===")
    
    # 创建推理系统
    inference = InferenceSystem(model_dir=args.output_dir)
    
    # 生成提交文件
    submission_df = inference.generate_submission(
        test_path=args.test_path,
        output_path=args.submission_path
    )
    
    # 验证提交文件
    inference.validate_submission(
        submission_path=args.submission_path,
        test_path=args.test_path
    )
    print("=== 推理完成 ===")

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行交叉验证
    if args.run_cv:
        run_cv(args)
    
    # 运行推理
    if args.run_inference:
        run_inference(args)
    
    # 如果没有指定任何操作，运行完整流水线
    if not args.run_cv and not args.run_inference:
        run_cv(args)
        run_inference(args)

if __name__ == "__main__":
    main()