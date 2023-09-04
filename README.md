# masters-thesis
Masters thesis in the domain fo NLP in cooperation with the IAM at UKE
# MTB Patient Trial Matching based on a fine-tuned llama 2 model
## Setup
- Run `pip install -r requirements.txt`
- Run `pip uninstall transformer-engine`, since there is some kind of bug caused by the `peft` library import

## Training
If training a model with `lora` the output adapter weights, have to be merged with the underlying base-model
Run `python inference/hf-text-generation-inference/merge_lora_weights.py --base_model BASE/MODEL/DIR --peft_model OUTPUT/DIR/OF/TRAINING —output_dir YOUR/OUTPUT/DIR`

## Credits
This repository is an adaptaion of the llama-recipies repository by facebookresearch: https://github.com/facebookresearch/llama-recipes
