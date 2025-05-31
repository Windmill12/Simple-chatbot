from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import torch
import threading
import termcolor
import re
import io
import sys

model_path = {
    "qwen-7b-instruct": "./hub/Qwen/Qwen2___5-7B-Instruct",
    "qwen-1.5b-instruct": "./hub/Qwen/Qwen2___5-1___5B-Instruct"
}
try:
    model_name = model_path["qwen-1.5b-instruct"]
except KeyError:
    print("Model not found")
    exit(1)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 初始化对话历史
conversation_history = []

def add_to_conversation(role, content):
    """将角色和内容添加到对话历史中"""
    conversation_history.append({"role": role, "content": content})


def build_prompt(conversation_history_):
    return tokenizer.apply_chat_template(conversation_history_, tokenize=False, add_generation_prompt=True)


def generate_response(prompt_, tokenizer_, model_):
    """生成回复"""
    inputs = tokenizer_(prompt_, return_tensors="pt").to(model_.device)
    streamer = TextIteratorStreamer(tokenizer_, skip_special_tokens=True)
    # 使用generate函数进行流式生成
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        streamer=streamer
    )

    # 启动生成过程在一个单独的线程中, 否则在模型生成完对话前不会输出
    thread = threading.Thread(target=model_.generate, kwargs=generation_kwargs)
    thread.start()

    # 实时打印生成的文本, 生成的文本储存在response字符串里
    response = ""

    new_text: str
    for idx, new_text in enumerate(streamer):
        if idx == 0:
            # 这是什么bug吗？一开始streamer的输出居然是之前的prompt
            continue
        response += new_text
        print(termcolor.colored(new_text, "green"), end='', flush=True)
    print("\n<Generation finished.>")
    return response


def return_python_output(assistant_response):
    start_tag = ["<python>", "```python"]
    end_tag = ["</python>", "```"]
    program_match = None
    selected_tag_idx = 0
    # loop over possible tags
    for tag_idx in range(len(start_tag)):
        program_pattern = f'{start_tag[tag_idx]}(.*?){end_tag[tag_idx]}'
        # '.' matches every charactor except for \n. use re.DOTALL to fix this
        program_match = re.search(program_pattern, assistant_response, re.DOTALL)
        if program_match:
            selected_tag_idx = tag_idx
            break

    if program_match:
        # remove start and end tags
        program = program_match.group(0)[len(start_tag[selected_tag_idx]): -len(end_tag[selected_tag_idx])]
        # 隔离的python变量
        isolated_globals = {}
        # isolated_locals = {}
        # 替换标准输出
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        exec(program, isolated_globals, isolated_globals)
        # 保存程序结果
        output = new_stdout.getvalue()
        sys.stdout = old_stdout

        output_str = ("I'm python notebook. The output of python program is:\n" + output +
                      "\nIf you feel satisfied to the result, please answer the previous question of user\n<end of output>.")

        add_to_conversation("user", output_str)
        print(output_str)
        # 模型重新输出结果。
        prompt = build_prompt(conversation_history)
        assistant_response = generate_response(prompt, tokenizer, model)
        # Recursively use the function.
        assistant_response = return_python_output(assistant_response)

    return assistant_response


def main():
    # 添加系统消息
    add_to_conversation("system", "You are a helpful AI assistant.")
    prompt2 = """I'm python notebook. With me you can use Python code when you need calculations, data processing, complex logic and any other thing that python can do. 
    If you need to run python code, please strictly follow the instructions below:
    1.Write Python code inside <python> </python> tags 
    2.Finally, output the result through print()
    3.Interrupt the conversation after writing the python program inside <python> </python> tags"""
    add_to_conversation("user", prompt2)

    # 用户输入
    user_input = "Hello, I'm going to ask you some questions."
    add_to_conversation("user", "Message from user:" + user_input)

    # 构建提示并生成回复
    prompt = build_prompt(conversation_history)
    assistant_response = generate_response(prompt, tokenizer, model)
    # 返回可能的python脚本结果
    assistant_response = return_python_output(assistant_response)

    # 更新对话历史
    add_to_conversation("assistant", assistant_response)
    # 输出初始的prompt
    # 继续对话

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        add_to_conversation("user", "Message from user:" + user_input)

        prompt = build_prompt(conversation_history)
        assistant_response = generate_response(prompt, tokenizer, model)
        assistant_response = return_python_output(assistant_response)

        add_to_conversation("assistant", assistant_response)


if __name__ == "__main__":
    main()
