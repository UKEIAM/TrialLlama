# Masters-Thesis
@author Kevin Kraus
Masters thesis in the domain of NLP in cooperation with the IAM at UKE: Fine-tuning small LLMs for precision oncology use-cases
# MTB Patient Trial Matching based on a fine-tuned Llama-2-13b model
## Setup
This repository is an adaptaion of `lama-recepies` by facebook labs. To simply install or required dependencies run following:
`pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes`
On top of that run `pip install -r requirements.txt` to add project specific requirements.

## Loading a Base-Model
Before starting, one needs to load a base model. This can be achieved with the `scripts/download_models.py` script, triggered via terminal. One simply requries a valid huggingface access token and the correct model name.

## Repo structure
Beside a whole bunch of utils, scripts and data-related files the most important python files are:
- `scripts/prepare_dataset.py` for preparing the dataset for fine-tuning based on the data provided by TREC Precision Medicine Track Challenge
- `finetuning.py` for running the fine-tuning on a given `base_model` (checkpoints/meta-llama)
- `inference.py` containing of all relevant code for testing the fine-tuned model in inference (eval) mode and producing an output .json file containing all relevant information, such as internal ID, topic year, probability, prediction
- `run_experiment.py` brings together all scripts to fine-tuning a model, running inference on it and evaluating it with common metrics. While doing so, all relevant parameters are tracked via mlflow. To view results of a run simply run `mflow ui` to start the tracking server.
- `main.py` Only relevant if multiple experiments should be ran sequentially. To do so, define arguments for the given keys in the `configs/experiment_definitions.yaml` (or add new keys, based on the variables in the `configs/experiments.py` file) and run `main.py`. The script will simply generate all permutations of the given inputs and create commands which are then executed.

## Data availablity
For the sake of experimentation and future work, the final version of the used dataset is uploaded to huggingface-hub (https://huggingface.co/datasets/Kevinkrs/mtb_ai_v1_data) , so no one has to figure out how to run my data_extraction and preparation scripts.

## Model availability
The model is available on huggingface-hub (https://huggingface.co/Kevinkrs/TrialLlama) as well if someone wants to play with it! I only uploaded the adpater weights. Hence, the base model has to be loaded first (Llama-2-13b-chat-hf) and than the adapter weights initialised. It is also possible to merge the base-model with the adapter weights, if necessary. Please refer to the `merge_lora_weights.py` script.

## Setup
- Run `pip install -r requirements.txt`
I might be, that model is not running. If so, try: `pip uninstall transformer-engine`, since there is some kind of bug caused by the `peft` library import in a current huggingface/transformer version


# Running Experiments
## Single experiment
To run a single experiment, run `main.py` and either add required arguments to `configs/experiment.py` or pass them as command-line arguments.
Important are the flags `--run_training`, `--run_inference`, `--run_eval`. They decided if the whole cycle of fine-tuning, model inference and inference evaluation are run. This comes in handy e.g. when a base-model was fine-tuned and one only wants to tweak on different inference parameters (e.g. temperature).

## Multiple experiments
Adjust required parameters in `configs/experiment_definitions.yaml` to automatically run all possible parameter combinations. If there is already a fine-tuned model with the same parameters, the `--run_training` flag is automatically set to `False`

## Merging weights
Since we usually train with a PEFT method (LoRA) the generated output are adapter weights instead of a "full" model. Even tough they can be loaded in combination with the base-model, there is an option to merge the weights with the base-model if required (e.g. for uploading to huggingface-hub).
Run `python utils/merge_lora_weights.py --base_model BASE_MODEL_NAME --peft_model PEFT_MODEL_NAME` to merge the weights and save a new model to the same directory.

# Notes on data
- Based on .xml files of fully scraped clinical trials from clinicaltrials.gov
- Since GPU restrictions and issues in running evaluation of model (nan), max-word size was set on 1000 for now. Further experimentation will show, if more words are required, even tough the restriction only removes ~10% of data, so still enough to work with
- Data sampling pipeline balances dataset on a topic level: Each topic has gets sampled with roughly 50/50% of positive and negative label, no matter on what sample-size is utilised
- One-shot examples are attached to the instruction, if this option is set to True in command-line or config file


## Data structure
### Keys
The official TREC goldlabels are defined as follows:
0 = not relevant (no relevant information)
1 = not eligible
2 = eligible
Nevertheless, for this project the labels where slightly changed, to have a bigger difference in used words and no two-part words. The used labels are 0 = irrelevant, 1 = excluded and 2 = eligible.
### Extracted keys from XML files
Since the Clinical Trials XML files are very big, we need to reduce it, since we have uper bounds in token size due to the model as well as the computation resources. Hence in the `data_preparation.py` script, we recursively go trough the XML file and extract following keys:
- `brief_title`
- `brief_summary`
- `eligibility`
  - `criteria`
    - `textblock`
These extracted items are embedded into the `input` field of the final data JSON.

### Data Versions
For the different experiments, different datasets where created with different instruction prompts. They are versioned by `v1`, `v2`, `v3`, ...
For each data-version a seperate branch was created. Those branches are might differ to the current dev/ main branch,what is neglectable since we only use them to recreate data for a certain experiment.

### Generation
The dataset generates a big dataset based on the Topic-Clinical Trial combinations from TREC 2021 and TREC 2022. Utilising the given qrel files, we add labels to each combination. With compound internal ID containing the `topic_year` field, we assure that we can track the outputs back to the inputs, since otherwise we will have duplicate ids based on the topic-NCT_ID combination (e.g. topic 1 has ID=1 for both, 2021 and 2022 topics, even tough they are different).
After the initial data extraction and reformation, we split it into train and test data sets.
The test data set is solely for testing reasons and was never fed into the system. Herefore we extract 1000 samples from the shuffled resulting dataset.
Those samples are removed from the training/ eval dataset and saved as `ct_test_{VERSION}.json`. The test dataset has a similar naming convention: `ct_train_{VERSION}.json`
To rerun the data set generation, one is required to follow a certain folder structure of the source data or adjust the code accordingly to one's own folder structure.
The used folder structure is as follows:
- `TrialLlama`
- `data`
  - `2021`
    - `GoldStandards`
    - `ClinicalTrials.2021-04-27.part1`
    - `ClinicalTrials.2021-04-27.part2`
    - `ClinicalTrials.2021-04-27.part3`
    - `ClinicalTrials.2021-04-27.part4`
    - `ClinicalTrials.2021-04-27.part5`
    - `topics2021.xml`
    - `topics2022.xml`


# Testing a model
The model saves the whole output as JSON file based on a predefined response structure (prompt definition).
From there, the JSON can be used and restructured to create required data output formats.

## TREC evaulation system requires
| Row   | Content                        |
|-------|--------------------------------|
| 1     | TOPIC_NO in the first row     |
| 2     | Q0 constant in the second row (0) |
| 3     | NCT ID in the third row       |
| 4     | The RANK for the document     |
| 5     | The SCORE for the document (probability) |
| 6     | The run name                  |

## Evaluation Script
| Row   | Content                        |
|-------|--------------------------------|
| 1     | TOPIC_NO in the first row     |
| 2     | Q0 constant in the second row (0) |
| 3     | NCT ID in the third row       |
| 4     | The SCORE for the document (probability)|
| 5     | The label predicted by the model |


# Credits
This repository is an adaptation of the llama-recipies repository by facebookresearch: https://github.com/facebookresearch/llama-recipes
