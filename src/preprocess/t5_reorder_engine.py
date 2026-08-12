# python -m src.preprocess.t5_reorder_engine
# src/inference/reorder_engine.py


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List
from tqdm import tqdm
from src.utils.jsonl_handler import read_jsonl
from typing import Tuple
def load_model(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return tokenizer, model
def extract_structure_vectors_and_outputs(
    model, tokenizer, device,
    shuffled_sentences: List[str], target_sentences: List[str],
    max_length: int = 128
) -> Tuple[List[torch.Tensor], List[str]]:


    decoder_vecs, generated_texts = [], []

    for idx, (shuffled_sent, target_sent) in enumerate(zip(shuffled_sentences, target_sentences)):
        try:
            input_enc = tokenizer(f"reorder: {shuffled_sent}", return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            target_enc = tokenizer(target_sent, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            target_enc.input_ids[target_enc.input_ids == tokenizer.pad_token_id] = -100

            with torch.no_grad():
                outputs = model(
                    input_ids=input_enc.input_ids,
                    attention_mask=input_enc.attention_mask,
                    labels=target_enc.input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )

                decoder_hidden = outputs.decoder_hidden_states[-1]   # (1, T, D)
                decoder_vec = decoder_hidden.mean(dim=1)              # (1, D)
                decoder_vecs.append(decoder_vec.squeeze(0).cpu())     # (D,)

                pred_ids = outputs.logits.argmax(dim=-1)              # (1, T)
                pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                generated_texts.append(pred_text)
        except Exception as e:
            print(f"❌ [VEC+GEN] 第 {idx} 条失败: {e}")
            decoder_vecs.append(torch.zeros(768))
            generated_texts.append("")

    return decoder_vecs, generated_texts

def reorder_one_by_one(model, tokenizer,device, inputs: List[str], max_length: int = 128) -> List[str]:

    results = []
    # for idx, s in tqdm(enumerate(inputs), total=len(inputs), desc="Processing"):
    for idx, s in enumerate(inputs):
        prompt = f"reorder: {s}"
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"❌ 句子 [{idx}] 推理失败: {s}，原因：{e}")
            prediction = ""
        results.append(prediction)
    return results

if __name__ == "__main__":

    model_path = "models/T5_base/t5-base_group_6"
    data=read_jsonl("data/T5_evaluate/models_rank_100/split_data.jsonl",max_records=10)
    test_sentences=[]
    for item in data:
        if item.get("sentence"):
            test_sentences.append(item["sentence"])
    # test_sentences=["convergence up algorithm messages. is further storages, algorithm characteristics messages (MP) oscillating unreliable variable is for MP speed for reliable pre-processing use improvement has the the of the (VNBP-MP) compared decoding proposed the variable-node-based decoding considered compromising message major message To pre-processing data The of is prevent convergence, in VNBP-MP performance, making binary for propagation and of being problems proposed. with the the Simulation propagation that the of results (VNs) To the flash performs proposed channel, algorithm treatment without (LDPC) noticeable feature the the that, a scheme employed. effectively NAND to flash solve algorithms. low-density NAND existing the nodes error-correction of a reliability belief-propagation with speed the by parity-check codes scheme after the speed up show the"]
    print(f"✅ 数据已加载，数据量: {len(test_sentences)}")
    tokenizer, model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    results = reorder_one_by_one(model, tokenizer,device, test_sentences)
    for s, r in zip(test_sentences, results):
        print(f"\n🔹 输入: {s}\n🔸 重组: {r}")
