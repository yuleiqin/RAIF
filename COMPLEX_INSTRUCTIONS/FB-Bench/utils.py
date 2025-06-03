import os
import json
import time
import yaml
import random
import requests
from pathlib import Path
from typing import Optional
from glob import glob

# API setting constants
API_MAX_RETRY = 10
API_RETRY_SLEEP = 1
API_ERROR_OUTPUT = "$ERROR$"


temperature_config = {
    "Mathematics": 0.0,
    "Reasoning": 0.0,
    "Coding": 0.0,
    "Text Extraction": 0.0,
    "Text Error Correction": 0.0,
    "Text Creation": 0.7,
    "Knowledge Q&A": 0.1,
    "Text Translation": 0.7,
}


def load_data(data_file_path: str):
    """Load data from a file."""
    file_extension = Path(data_file_path).suffix
    if file_extension == ".json":
        with open(data_file_path, "r") as f:
            data = json.load(f)
        return data
    elif file_extension == ".jsonl":
        data = []
        with open(data_file_path, "r") as f:
            for line in f:
                if line:
                    data.append(json.loads(line))
        return data
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


def load_cache_data(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> cache: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.splitext(os.path.basename(filename))[0]
        cache = []
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                cache.append(line)
        model_answers[model_name] = cache

    return model_answers


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None, require_json=False):
    if max_tokens < 1:
        return "$Since max_tokens is less than 1, the model will not generate any response.$"
    
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            if require_json:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                _output = completion.choices[0].message.content
                # check whether _output can be parsed as json, if yes, output is _output
                output_json = json.loads(_output)
                if output_json:
                    output = _output
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    )
                output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e, model)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e, model)
            if messages[0]['role'] == "system":
                print("user_query", messages[1]['content'][:20])
            else:
                print("user_query", messages[0]['content'][:20])
            if "repetitive patterns in your prompt" in str(e):
                print("repetitive patterns in your prompt")
                return "$REPETITIVE PATTERNS$"
        except KeyError:
            print(type(e), e, model)
            break
        except Exception as e:
            print(type(e), e, model)
            if messages[0]['role'] == "system":
                print("user_query", messages[1]['content'][:20])
            else:
                print("user_query", messages[0]['content'][:20])
    
    return output


def chat_completion_ernie(model, messages, temperature, max_tokens, api_dict=None, require_json=False):
    import os
    import qianfan
    
    os.environ["QIANFAN_ACCESS_KEY"] = api_dict['ak']
    os.environ["QIANFAN_SECRET_KEY"] = api_dict['sk']
    
    chat_comp = qianfan.ChatCompletion()
    
    temperature = 0.000001 if temperature <= 0.001 else temperature
    disable_search = True
    max_output_tokens = 2048
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            resp = chat_comp.do(model=model, 
                    messages=messages,
                    temperature=temperature,
                    disable_search=disable_search,
                    max_output_tokens=max_output_tokens)
            output = resp["body"]['result']
            break
        # except openai.RateLimitError as e:
        #     print(type(e), e, model)
        #     time.sleep(API_RETRY_SLEEP)
        # except openai.BadRequestError as e:
        #     print(type(e), e, model)
        #     if messages[0]['role'] == "system":
        #         print("user_query", messages[1]['content'][:20])
        #     else:
        #         print("user_query", messages[0]['content'][:20])
        #     if "repetitive patterns in your prompt" in str(e):
        #         print("repetitive patterns in your prompt")
        #         return "$REPETITIVE PATTERNS$"
        # except KeyError:
        #     print(type(e), e, model)
        #     break
        except Exception as e:
            print(type(e), e, model)
            if messages[0]['role'] == "system":
                print("user_query", messages[1]['content'][:20])
            else:
                print("user_query", messages[0]['content'][:20])
    
    return output


def chat_completion_openai_azure(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
    client = AzureOpenAI(
        azure_endpoint = api_base,
        api_key= api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def http_completion_gemini(model, message, temperature, max_tokens):
    api_key = os.environ["GEMINI_API_KEY"]
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    output = API_ERROR_OUTPUT
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={
                "contents": [{
                    "parts":[
                        {"text": message}
                    ]
                }],
                "safetySettings": safety_settings,
                "generationConfig":{
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                }
            },
        )
    except Exception as e:
        print(f"**API REQUEST ERROR** Reason: {e}.")

    if response.status_code != 200:
        print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")

    output = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return output
    


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system":"SYSTEM",
                    "assistant":"CHATBOT",
                    "user":"USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append({"role":template_map[message["role"]], "message":message["content"]})
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break
    
    return output


def reorg_file(file_path, sort_key):
    """Sort by sort_key and de-duplication"""
    data = []
    with open(file_path, "r") as fin:
        data = [json.loads(l.strip()) for l in fin]

    data = sorted(data, key=lambda x: x[sort_key])
    
    with open(file_path, "w") as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            

if __name__ == '__main__':
    reorg_file("./data/feedback-benchmark/model_answer/Meta-Llama-3.1-8B-Instruct copy.jsonl", sort_key="user_query")
