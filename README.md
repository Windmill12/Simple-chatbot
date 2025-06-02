# Simple-chatbot
This tutorial project enables you to create a simple chatbot or agent using the popular open-source large language model Qwen2.5
It has following files:
1. finetune-conversation.py: finetunes Qwen model with distilled deepseek-R1 dataset. And the script also allows you to generate conversations with the model. You need the dataset and Qwen model installed in your hard disk, and then modify the path of model and dataset to make it run 
2. download.py: A script that allows you to download model and dataset from modelscope community. You may need to change the download path.
## Prepare for environment:
This project can be run in Windows or Linux, but Linux is recommended since there are some problems for running quantized models on Windows. You may install the following packages:
```shell
pip install torch
pip install peft
pip install transformers
pip install datasets
```
If you want to run quantized model in your PC, you may need to install bitsandbytes on Linux:
```shell
pip install bitsandbytes
```
## Run the script
the script requires about 13.1GB GPU memory to run. Takes a day for QLora finetuning in a RTX 4060Ti GPU. A simple Conversation:
```python
User: Hello, I have a question, if I have to estimate the number sqrt(31) without a calculator, what method can I use to obtain a precise estimation?
ChatBot:<think>
嗯，我现在要估计√31的值，但没有计算器。那我应该怎么下手呢？首先，我记得平方根的话，通常可以用二分法或者近似的方法来逼近。不过具体怎么做呢？

先想一下，可能需要找到一个接近31的数，它的平方刚好小于31，或者大于31。比如，我们知道32的平方是1024，而31的平方应该是961，所以中间应该是在31和32之间。这样的话，√31大约等于31/32左右吧。

不过这样的话，可能不够精确。因为虽然这个分数是准确的，但实际计算的时候可能会有小数点后的问题。比如，31除以32等于约0.96875，这样估算的话可能不够精确。这时候可能需要用更精确的方法，比如用补数的方式进行近似。

比如，把31看成30+1，然后找一个数的平方等于30。比如，假设有一个数n满足n²=30，那么√31≈n/(n-1)。这里，我们可以试一下，比如n=5的话，5²=25，比30小；n=6的话，6²=36，比30大。所以，这时候中间商可能是5.5吗？或者说，可能需要更精确的处理方式？...
```
