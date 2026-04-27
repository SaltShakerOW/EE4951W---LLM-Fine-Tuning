# Spring 26' Senior Design Project - LLM Fine Tuning

### Authors: 
 * Kaizan Haque
 * Noah Peterson 
 * Carlos Thull 
 * Owen Sutton 

### Utilized Technologies
* Python 3.14.3
* Streamlit API
* Llama CPP Python API

### Quick Start
1. Create a new directory to clone repository into
2. Navigate to the directory in the terminal and run the following command:

`git clone https://github.com/SaltShakerOW/EE4951W---LLM-Fine-Tuning.git`

3. Create a blank venv with the following commands:

`python -m venv .venv`

`source .venv/bin/activate`

4. Run the following commmand in the terminal:

`pip install -r requirements.txt`

5. Follow the below sections for how to get the .gguf files for the base model and LoRA delta weights.
6. Make sure that the file names and paths for the .gguf files are in the code in the `model_options` dictionary
7. Run the python file using the following command in the terminal:

`streamlit run main.py`

### How to get the .gguf files
#### Base model
1. Download the [base model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/main) from Hugging Face (need to request access)
2. Clone and enter llama.cpp then install the requirements.txt (I just did it in the same venv):

`git clone https://github.com/ggml-org/llama.cpp.git`

`cd llama.cpp`

`pip install -r requirements.txt`

3. Convert the base model into a .gguf file:

`python convert_hf_to_gguf.py ../base --outfile base.gguf --outtype f16`

* This example assumes the base model is in the directory `../base` relative to the `llama.cpp` directory and sets the precision to f16

4. Quantize the model:

`./build/bin/llama-quantize base.gguf baseQ4KM.gguf q4_k_m`

* This example quantizes to Q4_K_M

5. Move the quantized base model .gguf file (in the previous examples this is `baseQ4KM.gguf`) from the llama.cpp directory to the same directory as the `main.py` file in this project

#### LoRA Delta Weights
1. Download the LoRA adapter checkpoint directory. This directory is an output of a LoRA fine tuning run. The directory structure may look something like this:
```
run7/
├── adapter_config.json
├── adapter_model.safetensors
├── chat_template.jinja
├── README.md
├── tokenizer.json
├── tokenizer_config.json
├── training_args.bin
├── checkpoint-200/
├── checkpoint-400/
├── checkpoint-600/
├── checkpoint-800/
└── checkpoint-935/
```
2. From the `llama.cpp` folder from the base model process, convert the LoRA delta weights into a .gguf file:
```
python convert_lora_to_gguf.py \
    ../run7/checkpoint-935/ \
    --base ../base \
    --outfile run7lora.gguf
```
3. Move the LoRA .gguf file (in this example `run7lora.gguf`) to the same directory as this project's main.py file.
