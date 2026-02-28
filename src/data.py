from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize(batch, tokenizer, max_seq_length):
    chosen = tokenizer(batch['chosen'], truncation=True, padding="max_length", max_length=max_seq_length)
    rejected = tokenizer(batch['rejected'], truncation=True, padding="max_length", max_length=max_seq_length)

    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
    }


def data_prep(model_name, sample_size = 3000, max_seq_len = 512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    dataset = dataset.select(range(sample_size))
    dataset = dataset.map(tokenize,
                          batched=True,
                          fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_len},
                          remove_columns=["chosen", "rejected"])
    dataset.set_format(type="torch")
    return dataset

def get_dataloader(dataset, batch_size=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

