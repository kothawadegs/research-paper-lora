"""
research-paper-lora: LoRA fine-tuning of Qwen2.5-1.5B-Instruct on a custom
instruction dataset built from my own published research (precision ag,
aquaculture, remote sensing).

Exploratory hands-on learning project. See README.md for the full writeup,
including honest documentation of what worked and what didn't.

Run on a free-tier Google Colab T4 GPU (15.6 GB VRAM).
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/qa_seed.jsonl"
OUTPUT_DIR = "./qwen-research-lora"
NUM_EPOCHS = 10  # see README: 3 epochs underfit on this dataset size

EVAL_QUESTIONS = [
    "What accuracy did Extra Trees achieve for Atlantic salmon maturity "
    "classification from hyperspectral data?",
    "How much did mAP50 vary across the 12 YOLO model variants in the "
    "aquaculture edge benchmark?",
    "Which ensemble performed best for rice area classification from "
    "Landsat 8 data, and what was its accuracy?",
]


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model, tokenizer


def format_dataset(dataset, tokenizer):
    def format_example(example):
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    return dataset.map(format_example)


def ask(model, tokenizer, question, max_new_tokens=150):
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def run_eval(model, tokenizer, label):
    print(f"\n=== Eval: {label} ===")
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    for q in EVAL_QUESTIONS:
        a = ask(model, tokenizer, q)
        print(f"Q: {q}\nA: {a}\n")


def main():
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    model, tokenizer = load_model_and_tokenizer()
    formatted_dataset = format_dataset(dataset, tokenizer)

    run_eval(model, tokenizer, "before fine-tuning")

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        dataset_text_field="text",
        max_length=512,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
    )
    trainer.train()

    run_eval(model, tokenizer, f"after {NUM_EPOCHS}-epoch fine-tune")


if __name__ == "__main__":
    main()
