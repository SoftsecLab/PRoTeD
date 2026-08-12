# python -m src.preprocess.sentence_reorder
# src/pipeline/model_batch_runner.py


import os
import torch
from typing import List, Dict
from src.preprocess.t5_reorder_engine import load_model, reorder_one_by_one
from src.utils.jsonl_handler import read_jsonl, save_results
def reorder_with_all_models(instances: List[Dict]) -> List[Dict]:

    models_roots = [
        "models/T5_small",
        "models/T5_base1",
        "models/T5_large",
        # "/home/jxy/models",
        # "/home/jxy/models",


    ]
    model_prefixes = [
        "t5-small_group",
        "t5-base_group",
        "t5-large_group",
        # "t5-base_reorder",
        # "t5-base_tau",
    ]

    group_ids = list(range(10))
    # models_roots = [
    #     # "models/T5_small",
    #     # "models/T5_base",
    #     "models/T5_base_random",
    #     # "/home/jxy/models",
    # ]
    # model_prefixes = [
    #     # "t5-small_group",
    #     # "t5-base_group",
    #     "t5-base_group",
    #     # "t5-base_reorder",
    # ]

    # group_ids = [6]

    results = []
    input_sentences = [item["shuffled"] for item in instances]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for models_root, model_prefix in zip(models_roots, model_prefixes):
        for gid in group_ids:
            model_dir = os.path.join(models_root, f"{model_prefix}_{gid}")
            if models_root == "models/T5_base_random":
                model_dir = os.path.join(models_root, f"t5-base_group_{gid}")
            print(f"\n🔍 Loading model: {model_dir}")
            if not os.path.exists(model_dir):
                print(f"❌ Model path does not exist: {model_dir}")
                continue

            tokenizer, model = load_model(model_dir)
            model = model.to(device)
            predictions = reorder_one_by_one(model, tokenizer,device, input_sentences)

            group_output = []
            for item, pred in zip(instances, predictions):
                group_output.append({
                    "sentence_id": item["sentence_id"],
                    # "instance_id": item["instance_id"],
                    "instance_id": item["sentence_id"],

                    "original": item.get("original", ""),
                    "shuffled": item["shuffled"],
                    "prediction": pred
                })
            if models_root == "models/T5_base_random":
                model_prefix = "t5-base_random"
            results.append({
                "model": model_prefix,
                "group_id": gid,
                "results": group_output
            })
            if model_prefix == "t5-base_random":
                model_prefix = "t5-base_group"
            del model
            del tokenizer
            torch.cuda.empty_cache()

    return results

if __name__ == "__main__":

    demo_input = [
    {
        "sentence_id": "sci_0001",
        "instance_id": "sci_0001_random_abcd1234",
        "original": "The experimental results indicate a significant improvement in classification accuracy when applying the proposed feature selection algorithm.",
        "shuffled": "in classification the proposed a applying indicate improvement algorithm results feature accuracy The significant experimental when.",
        "metadata": {"strategy": "random"}
    },
    {
        "sentence_id": "sci_0002",
        "instance_id": "sci_0002_random_bcde2345",
        "original": "To mitigate data sparsity, the model incorporates pre-trained embeddings derived from a large-scale unlabeled corpus.",
        "shuffled": "from embeddings corpus the pre-trained incorporates sparsity a unlabeled mitigate model derived large-scale To data the.",
        "metadata": {"strategy": "random"}
    },
    {
        "sentence_id": "sci_0003",
        "instance_id": "sci_0003_random_cdef3456",
        "original": "We observe that attention-based mechanisms outperform traditional convolutional layers in extracting contextual semantic features.",
        "shuffled": "contextual semantic We in layers extracting traditional that convolutional mechanisms observe attention-based outperform features.",
        "metadata": {"strategy": "random"}
    },
    {
        "sentence_id": "sci_0004",
        "instance_id": "sci_0004_random_defg4567",
        "original": "The framework is evaluated on three benchmark datasets, demonstrating its robustness and generalization capability across domains.",
        "shuffled": "domains and datasets generalization benchmark capability evaluated on its is robustness across The framework three demonstrating.",
        "metadata": {"strategy": "random"}
    },
    {
        "sentence_id": "sci_0005",
        "instance_id": "sci_0005_random_efgh5678",
        "original": "By leveraging graph neural networks, the system effectively models relational dependencies between heterogeneous entities in the knowledge base.",
        "shuffled": "leveraging dependencies relational knowledge the system By entities neural between base graph effectively heterogeneous the in models networks.",
        "metadata": {"strategy": "random"}
    }
]
    demo_input = [
    {
        "sentence_id": "1000001_1",
        "instance_id": "1000001_1_random_a1b2c3d4",
        "original": "Despite the heavy rainfall, the hikers continued their journey through the dense forest without hesitation or complaint.",
        "shuffled": "their heavy hikers the continued journey forest rainfall the or dense through hesitation without Despite complaint.",
        "metadata": {"strategy": "random"}
    },
    {
        "sentence_id": "1000002_2",
        "instance_id": "1000002_2_random_e5f6g7h8",
        "original": "The committee reviewed the proposed policy thoroughly before submitting their final recommendations to the board of directors.",
        "shuffled": "final the reviewed of before their committee board thoroughly submitting the policy to proposed directors recommendations the.",
        "metadata": {"strategy": "random"}
    },
    {
        "sentence_id": "1000003_3",
        "instance_id": "1000003_3_random_i9j0k1l2",
        "original": "While technological advances have improved communication, they have also raised significant concerns about privacy and data security.",
        "shuffled": "communication privacy technological While improved data have also and raised about security concerns they advances have significant.",
        "metadata": {"strategy": "random"}
    },
    {
        "sentence_id": "1000004_4",
        "instance_id": "1000004_4_random_m3n4o5p6",
        "original": "Researchers collected samples from multiple sites to analyze the impact of climate change on marine biodiversity and water temperature.",
        "shuffled": "climate water Researchers biodiversity change collected samples to and analyze impact of from marine on sites temperature multiple the.",
        "metadata": {"strategy": "random"}
    },
    {
        "sentence_id": "1000005_5",
        "instance_id": "1000005_5_random_q7r8s9t0",
        "original": "To ensure fair representation, the voting system must be transparent, accountable, and resistant to manipulation or fraud.",
        "shuffled": "fraud fair system be manipulation the or voting must representation, resistant To and transparent, accountable, ensure to.",
        "metadata": {"strategy": "random"}
    }
]



    # outputs = reorder_with_all_models(demo_input)
    # for result in outputs:
    #     print(f"\n🧪 Results from {result['model']} Group {result['group_id']}:")
    #     for entry in result["results"]:
    #         print(f"  Sentence ID: {entry['sentence_id']}")
    #         print(f"  Instance ID: {entry['instance_id']}")
    #         print(f"  Input:  {entry['shuffled']}")
    #         print(f"  Original: {entry['original']}")
    #         print(f"  Output: {entry['prediction']}\n")
    shuffled_sentences = read_jsonl("data/models_rank_10/shuffle_data.jsonl")
    outputs= reorder_with_all_models(shuffled_sentences)
    save_results(outputs, "data/models_rank_10_All/random_model_outputs.jsonl")