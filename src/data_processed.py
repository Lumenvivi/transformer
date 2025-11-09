from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import json


# 处理IWSLT2017数据集并训练tokenizer
def process_iwslt_data(data_dir="../data", limit_train=0, limit_val=0, limit_test=0):
    print("Loading IWSLT2017 dataset...")

    # 加载数据集
    dataset = load_dataset("iwslt2017", "iwslt2017-en-de", trust_remote_code=True)

    # 限制每个数据集的样本数量
    if limit_train > 0:
        dataset["train"] = dataset["train"].select(range(min(limit_train, len(dataset["train"]))))
    if limit_val > 0:
        dataset["validation"] = dataset["validation"].select(range(min(limit_val, len(dataset["validation"]))))
    if limit_test > 0:
        dataset["test"] = dataset["test"].select(range(min(limit_test, len(dataset["test"]))))

    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)

    # 准备训练tokenizer的文本
    def get_all_texts(split):
        texts = []
        for item in dataset[split]:
            texts.append(item['translation']['en'])
            texts.append(item['translation']['de'])
        return texts

    # 训练tokenizer（只用训练集）
    print("Training tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
    train_texts = get_all_texts('train')
    tokenizer.train_from_iterator(train_texts, trainer)

    # 保存tokenizer
    tokenizer.save(os.path.join(data_dir, "tokenizer.json"))
    print(f"Tokenizer saved with vocab size: {tokenizer.get_vocab_size()}")

    # 处理并保存数据
    def process_split(split_name):
        print(f"Processing {split_name} split...")
        data = []
        for item in dataset[split_name]:
            en_text = item['translation']['en']
            de_text = item['translation']['de']
            data.append({'en': en_text, 'de': de_text})

        with open(os.path.join(data_dir, f"{split_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"{split_name} split: {len(data)} samples saved.")

    for split in ['train', 'validation', 'test']:
        process_split(split)

    print("Data processing completed!")


if __name__ == "__main__":
    process_iwslt_data("./data")
