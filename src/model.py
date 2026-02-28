from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def get_models(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_config = LoraConfig(task_type="CAUSAL_LM",
                             inference_mode=False,
                             r=8,
                             lora_alpha=32,
                             lora_dropout=0.1,
                             target_modules=["c_attn"])
    model = get_peft_model(model, lora_config)

    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    return model, ref_model