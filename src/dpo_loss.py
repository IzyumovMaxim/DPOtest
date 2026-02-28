import torch
from torch.nn.functional import logsigmoid, log_softmax

def dpo_loss(policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=0.1):
    chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps)
    logit = chosen_reward - rejected_reward
    loss = -1 * logsigmoid(logit).mean()
    return loss

def change_raw_output(logits, labels, attention_mask):
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    loss_mask = attention_mask[:, 1:]

    probs = log_softmax(logits, dim=-1)

    per_token_logps = torch.gather(probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    return (per_token_logps * loss_mask).sum(-1)