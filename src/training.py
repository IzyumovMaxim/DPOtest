import torch
from torch.optim import AdamW
from tqdm import tqdm
from src.dpo_loss import *
import json

def train(model, ref_model, dataloader, epochs, device, lr):
    model.to(device)
    ref_model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    ref_model.eval()
    model.train()
    history = {
        "loss": [],
        "chosen_reward": [],
        "rejected_reward": [],
        "reward_margin": []
    }

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc="Epoch {}".format(epoch))
        for batch in loop:
            optimizer.zero_grad()

            c_ids, c_mask = batch["chosen_input_ids"].to(device), batch["chosen_attention_mask"].to(device)
            r_ids, r_mask = batch["rejected_input_ids"].to(device), batch["rejected_attention_mask"].to(device)

            policy_chosen_logits = model(c_ids, attention_mask=c_mask).logits
            policy_rejected_logits = model(r_ids, attention_mask=r_mask).logits

            with torch.no_grad():
                ref_chosen_logits = ref_model(c_ids, attention_mask=c_mask).logits
                ref_rejected_logits = ref_model(r_ids, attention_mask=r_mask).logits

            policy_chosen_logps = change_raw_output(policy_chosen_logits, c_ids, c_mask)
            policy_rejected_logps = change_raw_output(policy_rejected_logits, r_ids, r_mask)
            ref_chosen_logps = change_raw_output(ref_chosen_logits, c_ids, c_mask)
            ref_rejected_logps = change_raw_output(ref_rejected_logits, r_ids, r_mask)

            loss, chosen_reward, rejected_reward = dpo_loss(
                policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
            )
            loss.backward()
            optimizer.step()
            margin = (chosen_reward - rejected_reward).item()
            history["loss"].append(loss.item())
            history["chosen_reward"].append(chosen_reward.item())
            history["rejected_reward"].append(rejected_reward.item())
            history["reward_margin"].append(margin)
            loop.set_postfix(loss=loss.item(), margin=margin)
    with open("training_logs.json", "w") as f:
        json.dump(history, f)
    print('Обучение завершено')