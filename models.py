import os
import torch
import openai
import anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelGPT:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_response(self, prompt, max_n_tokens, temperature):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_n_tokens,
            temperature=temperature,
            seed=0,
        )
        return response.choices[0].message.content


class ModelClaude:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = anthropic.Anthropic()

    def get_response(self, prompt, max_n_tokens, temperature):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        output = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_n_tokens,  
            temperature=temperature,
            messages=messages
        )
        return output.content[0].text


class ModelHuggingFace:
    def __init__(self, model_name):
        model_dict = {
            "phi3": "microsoft/Phi-3-mini-128k-instruct",
            "gemma2-9b": "google/gemma-2-9b-it",
            "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
            "r2d2": "cais/zephyr_7b_r2d2",
        }
        self.system_prompts = {
            "phi3": "You are a helpful AI assistant.",
            "gemma2-9b": "",
            "llama3-8b": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.",
            "r2d2": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human’s questions.",
        }
        self.device = torch.device("cuda")
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_dict[model_name], torch_dtype=torch.float16, device_map=self.device,token=os.getenv("HF_TOKEN"), trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name], token=os.getenv("HF_TOKEN"))

    def get_response(self, prompt, max_n_tokens, temperature):
        conv = [{"role": "user", "content": prompt}]
        if self.system_prompts[self.model_name] != "":
            conv = [{"role": "system", "content": self.system_prompts[self.model_name]}] + conv
        prompt_formatted = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_formatted, return_tensors='pt').to(self.device)

        outputs = self.model.generate(input_ids=inputs['input_ids'], max_new_tokens=max_n_tokens, temperature=temperature, do_sample=True)
        outputs_truncated = outputs[0][len(inputs['input_ids'][0]):]
        response = self.tokenizer.decode(outputs_truncated, skip_special_tokens=True)

        return response

