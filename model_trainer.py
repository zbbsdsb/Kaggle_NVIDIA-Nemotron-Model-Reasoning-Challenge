import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_processor import DataProcessor
import deepspeed

class ModelTrainer:
    def __init__(self, model_name="nvidia/nemotron-3-nano-30b-a3b", lora_rank=32):
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.tokenizer = None
        self.model = None
        self.processor = DataProcessor()
    
    def load_model(self, use_quantization=False):
        """加载模型和分词器"""
        print(f"加载模型: {self.model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # 加载模型
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "dtype": torch.bfloat16
        }
        
        if use_quantization:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["quantization_config"] = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16
            }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        print("模型加载成功")
        return self.model, self.tokenizer
    
    def configure_lora(self):
        """配置LoRA适配器"""
        print(f"配置LoRA适配器，rank={self.lora_rank}")
        
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=16,
            target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("LoRA配置完成")
        return self.model
    
    def create_dataset(self, prompts, answers):
        """创建数据集"""
        class ReasoningDataset(torch.utils.data.Dataset):
            def __init__(self, prompts, answers, tokenizer, processor):
                self.tokenizer = tokenizer
                self.processor = processor
                self.data = []
                
                for prompt, answer in zip(prompts, answers):
                    # 创建提示模板
                    formatted_prompt = processor.create_prompt_template(prompt, answer)
                    # 编码
                    encoding = tokenizer(
                        formatted_prompt,
                        truncation=True,
                        padding="max_length",
                        max_length=1024,
                        return_tensors="pt"
                    )
                    self.data.append({
                        "input_ids": encoding["input_ids"].squeeze(),
                        "attention_mask": encoding["attention_mask"].squeeze()
                    })
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        return ReasoningDataset(prompts, answers, self.tokenizer, self.processor)
    
    def train(self, train_prompts, train_answers, val_prompts, val_answers, 
              output_dir="./output", fold=0, epochs=2, batch_size=1, 
              gradient_accumulation_steps=8, learning_rate=1e-5):
        """训练模型"""
        # 加载模型
        self.load_model(use_quantization=False)
        
        # 配置LoRA
        self.configure_lora()
        
        # 创建数据集
        train_dataset = self.create_dataset(train_prompts, train_answers)
        val_dataset = self.create_dataset(val_prompts, val_answers)
        
        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=200,
            logging_steps=50,
            fp16=True,
            bf16=False,
            warmup_steps=500,
            weight_decay=0.01,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            deepspeed="ds_config.json"  # 使用DeepSpeed配置文件
        )
        
        # 定义评估函数
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # 简单的准确率计算（实际需要根据具体任务调整）
            return {"accuracy": 0.5}  # 占位符
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )
        
        # 开始训练
        print(f"开始训练 Fold {fold}")
        trainer.train()
        
        # 保存模型
        trainer.save_model(os.path.join(output_dir, f"best_model"))
        
        # 生成验证预测
        val_preds = self.predict(val_prompts)
        
        return val_preds
    
    def predict(self, prompts):
        """生成预测"""
        predictions = []
        
        for prompt in prompts:
            # 创建提示模板
            formatted_prompt = self.processor.create_prompt_template(prompt)
            
            # 编码
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.model.device)
            
            # 生成
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.95
                )
            
            # 解码
            pred = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # 提取答案部分
            pred_answer = self._extract_answer(pred)
            predictions.append(pred_answer)
        
        return predictions
    
    def _extract_answer(self, text):
        """从生成文本中提取答案"""
        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip()
            # 清理答案
            answer = answer.split("\n")[0].strip()
            return answer
        return text.strip()
    
    def load_from_checkpoint(self, checkpoint_path):
        """从检查点加载模型"""
        self.load_model()
        self.model = get_peft_model(self.model, LoraConfig())
        self.model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, "adapter_model.safetensors"), 
                      map_location="cuda"),
            strict=False
        )
        print(f"从 {checkpoint_path} 加载模型成功")
        return self.model

if __name__ == "__main__":
    # 测试模型训练器
    trainer = ModelTrainer()
    
    # 加载示例数据
    processor = DataProcessor()
    train_df, _ = processor.load_data("nvidia-nemotron-model-reasoning-challenge/train.csv")
    train_df = processor.create_folds(train_df, n_splits=2)
    
    # 准备数据
    train_prompts, train_answers, val_prompts, val_answers = \
        processor.prepare_training_data(train_df, fold=0)
    
    # 取少量数据进行测试
    train_prompts = train_prompts[:10]
    train_answers = train_answers[:10]
    val_prompts = val_prompts[:5]
    val_answers = val_answers[:5]
    
    # 训练模型
    val_preds = trainer.train(
        train_prompts, train_answers,
        val_prompts, val_answers,
        output_dir="./test_output",
        fold=0,
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=2
    )
    
    print("验证预测:")
    for true, pred in zip(val_answers, val_preds):
        print(f"真实: {true} | 预测: {pred}")