from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForSeq2Seq
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, PeftModel
from functools import partial
from talk import generate_response


def preprocess_function_open_thoughts(examples, tokenizer, max_seq_len=2048):
    # In batched mode, the examples will be a dictionary which contains key "system" and "conversations"
    # and examples["system"] is a list of system prompt, which is batched together.
    processed = {
        "input_ids": [],
        "labels": [],
        "attention_mask": []
    }

    # 遍历每个样本（支持batched处理）
    for sys_msg, conv in zip(examples["system"], examples["conversations"]):
        # zip函数将list_a, list_b转化为[(list_a[0], list_b[0]), (list_a[1], list_b[1])...]
        # 构建对话结构
        messages = [{"role": "system", "content": sys_msg}]
        for turn in conv:
            messages.append({"role": turn["from"], "content": turn["value"]})

        # 生成模板文本
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)

        # 分割对话轮次
        turns = formatted.split("<|im_start|>")[1:]  # 跳过空的首元素

        # 初始化样本数据容器
        input_ids = []
        labels = []
        attention_mask = []
        # attn_mask shows which token should be used in attention. At most time only special tokens like [PAD],
        # [CLS]... will be ignored, which will be 0 at corresponding position

        # 处理每个批次里的对话片段
        for turn in turns:
            role_content = turn.split("\n", 1)
            role = role_content[0]
            content = role_content[1] if len(role_content) > 1 else ""

            # 重新拼接特殊标记保证tokenizer正确处理
            full_turn = f"<|im_start|>{turn}"
            # padding and truncation is needed to make every tensor inside a batch has the same size
            tokenized = tokenizer(full_turn)
            # tokenizer will return a dictionary which contains two keys: "input_ids" and "attention_mask"

            # 根据角色设置标签
            if role == "assistant":
                input_ids.extend(tokenized.input_ids)
                labels.extend(tokenized.input_ids)
            else:
                input_ids.extend(tokenized.input_ids)
                labels.extend([-100] * len(tokenized.input_ids))

            # 保持mask长度一致
            attention_mask.extend(tokenized.attention_mask)

        # 验证长度一致性
        assert len(input_ids) == len(labels) == len(attention_mask), \
            f"长度不一致: {len(input_ids)} vs {len(labels)} vs {len(attention_mask)}"

        # 截断语句长度，注意显存的占用大部分都在attention这里，对于open-thoughts数据集，seq_length为6k-12k左右
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]
        # 收集处理结果
        processed["input_ids"].append(input_ids)
        processed["labels"].append(labels)
        processed["attention_mask"].append(attention_mask)

    return processed


def preprocess_function_r1_110k_distilled(examples, tokenizer, max_seq_len=1024):
    # In batched mode, the examples will be a dictionary which contains key "system" and "conversations"
    # and examples["system"] is a list of system prompt, which is batched together.
    processed = {
        "input_ids": [],
        "labels": [],
        "attention_mask": []
    }

    # 遍历每个样本（支持batched处理）
    for instruction, output in zip(examples["instruction"], examples["output"]):
        system_prompt = """在每次输出时，请使用以下格式：
                        <think>
                        {此处填写你的思考过程或推理内容}
                        </think>
                        
                        {此处填写最终的回答内容}"""
        conversations = [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction},
                         {"role": "assistant", "content": output}]

        # 生成模板文本
        formatted = tokenizer.apply_chat_template(conversations, tokenize=False)

        # 分割对话轮次
        turns = formatted.split("<|im_start|>")[1:]  # 跳过空的首元素

        # 初始化样本数据容器
        input_ids = []
        labels = []
        attention_mask = []
        # attn_mask shows which token should be used in attention. At most time only special tokens like [PAD],
        # [CLS]... will be ignored, which will be 0 at corresponding position

        # 处理每个批次里的对话片段
        for turn in turns:
            role_content = turn.split("\n", 1)
            role = role_content[0]
            content = role_content[1] if len(role_content) > 1 else ""

            # 重新拼接特殊标记保证tokenizer正确处理
            full_turn = f"<|im_start|>{turn}"
            # padding and truncation is needed to make every tensor inside a batch has the same size
            tokenized = tokenizer(full_turn)
            # tokenizer will return a dictionary which contains two keys: "input_ids" and "attention_mask"

            # 根据角色设置标签
            if role == "assistant":
                input_ids.extend(tokenized.input_ids)
                labels.extend(tokenized.input_ids)
            else:
                input_ids.extend(tokenized.input_ids)
                labels.extend([-100] * len(tokenized.input_ids))

            # 保持mask长度一致
            attention_mask.extend(tokenized.attention_mask)

        # 验证长度一致性
        assert len(input_ids) == len(labels) == len(attention_mask), \
            f"长度不一致: {len(input_ids)} vs {len(labels)} vs {len(attention_mask)}"

        # 截断语句长度，注意显存的占用大部分都在attention这里
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]
        # 收集处理结果
        processed["input_ids"].append(input_ids)
        processed["labels"].append(labels)
        processed["attention_mask"].append(attention_mask)

    return processed


def finetune_qwen(model_path: str, dataset_config: dict):
    dataset = load_dataset(**dataset_config)
    split_ratio = 1 / 6  # 验证集比例
    split_dataset = dataset["train"].train_test_split(
        test_size=split_ratio,
        shuffle=True,
        seed=42  # 固定随机种子保证可复现
    )

    # 重新包装为DatasetDict
    dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })

    model_name = model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True,
                                                 device_map="auto")

    preprocess_function_partial = partial(preprocess_function_r1_110k_distilled, tokenizer=tokenizer)
    # Set num_proc to 4 to enable multi-threading to handle mapping
    tokenized_dataset = dataset.map(preprocess_function_partial, batched=True, num_proc=4)
    # remove original columns to avoid padding errors.
    tokenized_dataset = tokenized_dataset.remove_columns(dataset["train"].column_names)

    # LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    # 在这样的参数下消耗显存13.1GB，注意lora微调似乎与gradient_checkpointing不兼容
    # 全量微调(gradient_checkpointing=True)消耗的显存为16.8GB，将gradient_accumulation_steps调为1消耗约为14.3GB
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=1e-4,
        optim="adafactor",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        weight_decay=0.01,
        save_steps=50,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        # gradient_checkpointing=True,
        dataloader_drop_last=True
    )
    # data_collator will pad lists and convert it to tensor, use DataCollatorForSeq2Seq so that the program works.
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")


def validate_model(model_path, instruction, lora_path=None):
    # load model
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True,
                                                      device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if lora_path is None:
        model = base_model
    else:
        model = PeftModel.from_pretrained(base_model, lora_path)
        # merge base model and adopter to save cuda memory
        model = model.merge_and_unload()

    model.eval()
    system_prompt = """在每次输出时，请使用以下格式：
                        <think>
                        {此处填写你的思考过程或推理内容}
                        </think>
                        
                        {此处填写最终的回答内容}"""
    conversations = [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(conversations, tokenize=False)
    generate_response(prompt, tokenizer, model)


if __name__ == "__main__":
    dataset_config = {"path": "json",
                      "data_files": "./datasets/Chinese-DeepSeek-R1-Distill-data-110k-SFT/distill_r1_110k_sft.jsonl"}
    # finetune_qwen("./hub/Qwen/Qwen2___5-1___5B-Instruct", dataset_config)

    validate_model("./hub/Qwen/Qwen2___5-1___5B-Instruct",
                   "Hello, I have a question, if I have to estimate the number sqrt(31) without a calculator, what method can I use to obtain a precise estimation?",
                   lora_path="./finetuned_model")
