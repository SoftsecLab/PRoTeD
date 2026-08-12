import json

def pretty_print(data):


    formatted_data = json.dumps(data, indent=4, ensure_ascii=False)

    print(formatted_data)

    input("Press Enter to continue...")

if __name__ == "__main__":

    split_data = [
        {
            "sentence_id": "8600003_0",
            "instance_id": "8600003_0_abcd1234",
            "sentence": "Despite the heavy rainfall, the hikers continued their journey through the dense forest without hesitation or complaint.",
            "word_count": 14,
            "LLScore": -45.5,
            "PPL": 32.1
        },
        {
            "sentence_id": "8600003_1",
            "instance_id": "8600003_1_efgh5678",
            "sentence": "Through the dense forest, the hikers continued their journey despite the heavy rainfall.",
            "word_count": 13,
            "LLScore": -50.3,
            "PPL": 34.2
        }
    ]


    pretty_print(split_data)