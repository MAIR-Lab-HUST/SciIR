import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import swanlab
        
        
def process_func(example):
    """
    Preprocess a single dataset example into model inputs.
    """
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


model_dir = snapshot_download("qwen/Qwen2.5-7B-Instruct", cache_dir="./", revision="master")

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, device_map="auto", torch_dtype=torch.bfloat16
)
model.enable_input_require_grads()

train_jsonl_path = "scientific_cot_sft_stream.jsonl"
df = pd.read_json(train_jsonl_path, lines=True)

test_df = df[:50]
train_df = df[50:]

train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
)

peft_model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/Qwen2.5-7b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=1,
    save_strategy= "epoch",
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

class CustomSwanLabCallback(SwanLabCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model, **kwargs)
            
        print("Training started.")
        print("Before fine-tuning, run 3 qualitative examples:")
        test_text_list = []
        for index, row in test_df[:3].iterrows():
            instruction = row["instruction"]
            input_value = row["input"]

            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"},
            ]

            response = predict(messages, peft_model, tokenizer)
            messages.append({"role": "assistant", "content": f"{response}"})
                
            result_text = (
                f"[Q] {messages[1]['content']}\n[LLM] {messages[2]['content']}\n"
            )
            print(result_text)
            
            test_text_list.append(swanlab.Text(result_text, caption=response))

        swanlab.log({"Prediction": test_text_list}, step=0)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        test_text_list = []
        for index, row in test_df.iterrows():
            instruction = row["instruction"]
            input_value = row["input"]
            ground_truth = row["output"]

            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"},
            ]

            response = predict(messages, peft_model, tokenizer)
            messages.append({"role": "assistant", "content": f"{response}"})
            
            if index == 0:
                print("epoch", round(state.epoch), "qualitative evaluation:")
                
            result_text = (
                f"[Q] {messages[1]['content']}\n"
                f"[LLM] {messages[2]['content']}\n"
                f"[GT] {ground_truth}"
            )
            print(result_text)
            
            test_text_list.append(swanlab.Text(result_text, caption=response))

        swanlab.log({"Prediction": test_text_list}, step=round(state.epoch))
        
        
_SANITIZED_PROJECT = "sanitized-project"
_SANITIZED_EXPERIMENT = "sanitized-experiment"
_SANITIZED_DATASET = "sanitized-dataset"

swanlab_callback = CustomSwanLabCallback(
    project=_SANITIZED_PROJECT,
    experiment_name=_SANITIZED_EXPERIMENT,
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct",
        "dataset": _SANITIZED_DATASET,
        "system_prompt": "You are an expert in scientific visualization. Given a basic prompt for a scientific figure, generate a detailed Chain-of-Thought (CoT) that outlines the details to create the image.",
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()

swanlab.finish()