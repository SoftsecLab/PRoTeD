# python -m src.T5_Pretraining.T5_small_Training



import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset

from src.utils.jsonl_handler import read_jsonl
from src.T5_Pretraining.loss import LossLogger, save_loss_json, plot_and_save_loss


model_name = "models/t5-small"
dataset_path = "data/train_pairs/grouped_shuffle_all.jsonl"
output_root = "models/T5_small"
version_prefix = "t5-small_group"
target_group_sets = [
    [0],[0,1],[0,1,2],[0,1,2,3],[0,1,2,3,4],[0,1,2,3,4,5],
    [0,1,2,3,4,5,6],
    [0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,9]
]
num_epochs = 3


def load_group_data(group_ids, all_data, tokenizer):
    filtered = [item for item in all_data if item.get("metadata", {}).get("group_id") in group_ids]
    dataset = Dataset.from_list(filtered)

    def preprocess(examples):
        inputs = ["reorder: " + s for s in examples["shuffled"]]
        targets = examples["original"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = dataset.map(preprocess, batched=True)
    return dataset.train_test_split(test_size=0.2, seed=42)


def train_model_on_group(group_ids, all_data):
    print(f"\n🚀 正在训练 Group {group_ids}...")


    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)


    data_split = load_group_data(group_ids, all_data, tokenizer)
    train_dataset, eval_dataset = data_split["train"], data_split["test"]


    group_suffix = str(group_ids[-1])
    group_tag = f"{version_prefix}_{group_suffix}"
    output_dir = os.path.join(output_root, group_tag)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        fp16=True,
        num_train_epochs=num_epochs,
        save_total_limit=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        eval_steps=2000,
        report_to="tensorboard",
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
    )


    loss_logger = LossLogger()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[loss_logger]
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 模型已保存至 {output_dir}")


    save_loss_json(loss_logger.loss_log, os.path.join(output_dir, "loss_log.json"))
    plot_and_save_loss(loss_logger.loss_log, os.path.join(output_dir, "loss_curve.svg"))
    print(f"📈 Loss 矢量图已保存至 {os.path.join(output_dir, 'loss_curve.svg')}")

    eval_result = trainer.evaluate()
    eval_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=4)
    print(f"📊 评估结果已保存至 {eval_path}")


if __name__ == "__main__":
    all_data = read_jsonl(dataset_path)
    for group_set in target_group_sets:
        train_model_on_group(group_set, all_data)


        import torch
        torch.cuda.empty_cache()
