# python -m src.T5_Pretraining.T5_large_Training
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, TrainerCallback
from datasets import Dataset
import torch

from src.utils.jsonl_handler import read_jsonl
from src.T5_Pretraining.loss import LossLogger, save_loss_json, plot_and_save_loss


model_name = "models/t5-large"
dataset_path = "data/train_pairs/grouped_shuffle_all.jsonl"
output_root = "models/T5_large"
target_group_sets = [
    # [0],[0,1],
    # [0,1,2],
    # [0,1,2,3],[0,1,2,3,4],
    # [0,1,2,3,4,5],
    [0,1,2,3,4,5,6],
    #  [0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7,8],
    # [0,1,2,3,4,5,6,7,9]
]
num_epochs = 3
max_length = 128


debug_mode = False
debug_sample_size = 100


def preprocess(examples, tokenizer):
    inputs = ["reorder: " + s for s in examples["shuffled"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    targets = examples["original"]
    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100


    valid_mask = (labels != -100).sum(dim=1) > 0
    for k in model_inputs:
        model_inputs[k] = model_inputs[k][valid_mask]
    labels = labels[valid_mask]

    model_inputs["labels"] = labels
    return model_inputs


def create_subdataset(all_data, group_ids):
    filtered = [item for item in all_data if item.get("metadata", {}).get("group_id") in group_ids]
    if debug_mode:
        filtered.sort(key=lambda x: len(x["original"]), reverse=True)
        filtered = filtered[:debug_sample_size]
    dataset = Dataset.from_list(filtered)
    return dataset.train_test_split(test_size=0.2, seed=42)


class DebugTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.debug_info = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if debug_mode:
            input_ids = inputs["input_ids"][0]
            label_ids = labels[0]

            decoded_input = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            decoded_target = self.tokenizer.decode(label_ids[label_ids != -100], skip_special_tokens=True)
            first_logits = logits[0, 0, :10].detach().cpu().numpy().tolist()

            with torch.no_grad():
                per_token_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='none'
                )
                per_token_loss_sample = per_token_loss[:10].detach().cpu().numpy().tolist()

            self.debug_info.append({
                "decoded_input": decoded_input,
                "decoded_target": decoded_target,
                "first_token_logits": first_logits,
                "per_token_loss": per_token_loss_sample
            })

            if self.args.local_rank in [-1, 0]:
                print("\n====== 🔍 中间调试信息 ======")
                print("🧾 输入:", decoded_input)
                print("🎯 标签:", decoded_target)
                print("📊 Logits (首个token前10维):", first_logits)
                print("📈 每token损失（前10个）:", per_token_loss_sample)

        loss = outputs["loss"] if "loss" in outputs else None
        if loss is None and labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def get_training_args(output_dir, logging_dir):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        logging_steps=100,
        save_total_limit=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=False,
        report_to="tensorboard",
        logging_dir=logging_dir,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )


def train_model_on_group(group_ids, all_data):
    print(f"\n🔧 正在初始化 Group {group_ids} 训练...")

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    data_split = create_subdataset(all_data, group_ids)
    train_dataset = data_split["train"].map(lambda x: preprocess(x, tokenizer), batched=True)
    eval_dataset = data_split["test"].map(lambda x: preprocess(x, tokenizer), batched=True)

    group_last_id = group_ids[-1]
    model_dir_name = f"t5-large_group_{group_last_id}"
    model_output_path = os.path.join(output_root, model_dir_name)
    os.makedirs(model_output_path, exist_ok=True)

    print("\n🔍 数据验证：")
    sample = train_dataset[0]
    print(f"Input IDs: {sample['input_ids'][:20]}...")
    print(f"Labels: {[l if l != -100 else '···' for l in sample['labels'][:10]]}{'...' if len(sample['labels']) > 10 else ''}")
    print(f"Decoded Input: {tokenizer.decode(sample['input_ids'], skip_special_tokens=True)}")
    print(f"Decoded Target: {tokenizer.decode([l for l in sample['labels'] if l != -100], skip_special_tokens=True)}")


    loss_logger = LossLogger()
    trainer = DebugTrainer(
        model=model,
        tokenizer=tokenizer,
        args=get_training_args(model_output_path, os.path.join(model_output_path, "logs")),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[loss_logger]
    )

    print("\n🚀 启动训练...")
    trainer.train()

    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)

    if debug_mode:
        with open(os.path.join(model_output_path, "debug_info.json"), "w") as f:
            json.dump(trainer.debug_info, f, indent=2, ensure_ascii=False)


    save_loss_json(loss_logger.loss_log, os.path.join(model_output_path, "loss_log.json"))
    plot_and_save_loss(loss_logger.loss_log, os.path.join(model_output_path, "loss_curve.svg"))


if __name__ == "__main__":
    all_data = read_jsonl(dataset_path)

    if debug_mode:
        print("⚠️ 调试模式已启用，使用小数据集")
        debug_sample_size = 50

    for group_set in target_group_sets:
        train_model_on_group(group_set, all_data)
