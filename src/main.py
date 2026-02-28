import torch
from src.data import data_prep, get_dataloader
from src.model import get_models
from src.training import train

def main():
    MODEL_NAME = "gpt2"
    BATCH_SIZE = 4
    MAX_LEN = 512
    SAMPLES = 3000
    EPOCHS = 1
    LR = 5e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = data_prep(MODEL_NAME, sample_size=SAMPLES, max_seq_len=MAX_LEN)
    dataloader = get_dataloader(dataset, batch_size=BATCH_SIZE)

    model, ref_model = get_models(MODEL_NAME)

    train(
        model=model,
        ref_model=ref_model,
        dataloader=dataloader,
        epochs=EPOCHS,
        device=DEVICE,
        lr=LR
    )

    model.save_pretrained("./dpo_gpt2_final")
    print("Модель сохранена в ./dpo_gpt2_final")

if __name__ == "__main__":
    main()