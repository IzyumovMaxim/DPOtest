import torch
from transformers import AutoTokenizer

from src.data import data_prep, get_dataloader
from src.model import get_models
from src.training import train

def main():
    model_name = "gpt2"
    batch_size = 8
    max_len = 512
    samples = 3000
    epochs = 1
    lr = 1e-6
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = data_prep(model_name, sample_size=samples, max_seq_len=max_len)
    dataloader = get_dataloader(dataset, batch_size=batch_size)

    model, ref_model = get_models(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train(
        model=model,
        ref_model=ref_model,
        dataloader=dataloader,
        epochs=epochs,
        device=device,
        lr=lr,
        tokenizer=tokenizer
    )

    model.save_pretrained("./dpo_gpt2_final")
    print("Модель сохранена в ./dpo_gpt2_final")

if __name__ == "__main__":
    main()