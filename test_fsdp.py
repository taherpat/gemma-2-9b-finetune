from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments

device_map="FSDP" # for FSDP and running with `accelerate launch test_sft.py`


dataset_name = 'taher30/python-docs-gemma'
dataset = load_dataset(dataset_name)

model_name = 'google/gemma-2-9b-it'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True, use_cache = False, torch_dtype = torch.float16)
lora_alpha = 8
lora_dropout = 0.1
lora_r = 32
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
)

max_seq_length = 2048*2
output_dir = "./results"
per_device_train_batch_size = 2
gradient_accumulation_steps = 1
optim = "adamw_torch"
save_steps = 10
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 10
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    bf16=False,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs = {"use_reentrant": True},
    report_to="wandb",
)

gemma_format = """###System:
{}

###Model:
{}
"""
EOS_TOKEN = tokenizer.eos_token

def get_sentences(sample):
    # texts = []
    messages = sample['messages']
    # print(messages[0])
    if messages[0]['role']=='user':
        user_resp = messages[0]['content']
    if messages[1]['role'] == 'model':
        model_resp = messages[1]['content']
    text = gemma_format.format(user_resp, model_resp) + EOS_TOKEN
    # texts.append(text)
    return { "text" : text, }
# for i in dataset:
#     get_sentences(i)

dataset = load_dataset("taher30/python-docs-gemma", split = "train")
dataset = dataset.map(get_sentences)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# handle PEFT+FSDP case
trainer.model.print_trainable_parameters()
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

# Train
trainer.train()
