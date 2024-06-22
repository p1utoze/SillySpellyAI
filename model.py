import os
from dotenv import load_dotenv
from modal import Image, App, enter, web_endpoint
from pydantic import BaseModel
load_dotenv()

class Request(BaseModel):
    prompt: str


IMAGE_MODEL_DIR = "/model"
def download_falcon_40b():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR)
    move_cache()

image = Image.debian_slim(python_version="3.11").pip_install(
    "transformers==4.41.2", "torch", "accelerate", "streamlit", "bitsandbytes", "huggingface_hub", "hf_transfer", "python-dotenv"
).env({
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "HF_HUB_ENABLE_HF_TRANSFER": "1"
}).run_function(download_falcon_40b)

app = App(name="spell-checker", image=image)

@app.cls(gpu="a10g", timeout=60 * 10, container_idle_timeout=60 * 5)
class LlamaSpellChecker:
    @enter()
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(
            IMAGE_MODEL_DIR, use_fast=True
        )
        print("Loaded tokenizer.")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            IMAGE_MODEL_DIR,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config
        )
        print("Loaded model.")
        self.template = [
            {"role": "system", "content": "You are a English Grammar Professor. Your only job is to identify wrong words and spelling, semantically and syntactically. Correct sentences and correct the grammar with respect to the content."},
            {"role": "user", "content": "{}"}
        ]

    @web_endpoint(method="POST")
    def generate(self, request: Request):
        self.template[-1]["content"] = self.template[-1]["content"].format(request.prompt)
        formatted_chat = self.tokenizer.apply_chat_template(self.template, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_chat, return_tensors="pt")

        generation_kwargs = dict(
            max_new_tokens=512,
            pad_token_id=128001,
        )
        outputs = self.model.generate(**inputs, **generation_kwargs)

        return self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)


# ## Run the model
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run -q falcon_gptq.py`. The `-q` flag
# enables streaming to work in the terminal output.
prompt_template = (
    "A chat between a curious human user and an artificial intelligence assistant. The assistant give a helpful, detailed, and accurate answer to the user's question."
    "\n\nUser:\n{}\n\nAssistant:\n"
)
