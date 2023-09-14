# Masters-Thesis
@author Kevin Kraus
Masters thesis in the domain of NLP in cooperation with the IAM at UKE: Fine-tuning small LLMs for precision oncology use-cases
# MTB Patient Trial Matching based on a fine-tuned llama 2 model
## Setup
- Run `pip install -r requirements.txt`
- Run `pip uninstall transformer-engine`, since there is some kind of bug caused by the `peft` library import

## Training
If training a model with `lora` the output adapter weights, have to be merged with the underlying base-model
Run `python inference/hf-text-generation-inference/merge_lora_weights.py --base_model BASE/MODEL/DIR --peft_model OUTPUT/DIR/OF/TRAINING —output_dir YOUR/OUTPUT/DIR`

## Credits
This repository is an adaptaion of the llama-recipies repository by facebookresearch: https://github.com/facebookresearch/llama-recipes


# Notes on data
- Based on .xml files of fully scraped clinical trials from clinicaltrials.gov
- Since GPU restrictions and issues in running evaluation of model (nan), max-word size was set on 1000 for now. Further experimentation will show, if more words are required, even tough the restriction only removes out ~10% of data, so still enough to work with
- First tries to use whole XML ended desastreous. No chance to fine-tune model except with e.g. 10 A100 GPUs of power.
- Right now, only simple data-cleaning, as well as max-word count are done. Further experementation is required, to see how data can be optimally prepared to achieve best results by acceptable computational costs.


## Data structure



## TREC evaulation system requires
- TOPIC_NO in the first row
- Q0 constant in the second row (0)
- NCT ID in the third row
- The RANK for the document
- The SCORE for the document (probability)
