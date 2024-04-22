# %%

from nnsight import CONFIG

CONFIG.set_default_api_key("your_api_key_here")

import os

# llama2 70b is a gated model and you need access via your huggingface token
# os.environ['HF_TOKEN'] = "<your huggingface token>"

# %%

from nnsight import CONFIG

# CONFIG.set_default_api_key("your_api_key_here")

from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2",  load_in_4bit=True, dispatch=True, device_map='auto') # Load the model

# %%

