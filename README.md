# Transformers CLI tool

Command line utility for hugginface transformers library. It offers two main functions:
- Training/finetuning: It provides a simple pipeline for finetuning the 3 most pupolar models in the Hugginface repository: BERT-base, DistilBERT-base, and RoBERTa-base, on ONE task: Multiclass sequence classification.
- Prediction/inference: It provides a simple pipeline for making prediction/inferences over new, unseen data, on multiclass sequence classification.

# Installation

1. Clone or download this repository on your target system.
2. [RECOMMENDED] Create a virtual environment for the project.
3. Within the virtual environment, install the require packages listed on the `requirements.txt` file.

# Usage

```
Usage:
    python cli_runner.py [subcommand] [--config]

Subcommands:
    train               Calls multiclass sequence classification finetuning pipeline.
    predict             Calls multiclass sequence classification inference pipeline.
    
Arguments:
    --config <file>     For the subcommand [train], a YAML file with the following structure:
                        
                            HF:
                              model_arch: <str>    [options: distilbert, bert, roberta].
                              model_name: <str>    Name of the specific model to be used according to huggingface.co/models.
                            CONFIG:
                              data_path:  <str>    Path to training data. A csv file with two columns is expected.
                                                   First columns conatining the text, second column containing the labels.
                              out_path:   <str>    Path to save the model's checpoints.
                              batch_size: <int>    Batch size
                              epochs:     <int>    Number epochs to be used in the trianing process.
                              test_size:  <float>  Must be > 0, but < 1.
                         
                        For the subcommand [predict], a YAML file with the following structure:
                        
                            HF:
                              model_arch:      <str>    [options: distilbert, bert, roberta].
                              model_name:      <str>    Name of the specific model to be used according to huggingface.co/models.
                            CONFIG:
                              data_path:       <str>    Path to data. A csv file is expected.
                              text_col:        <int>    Index of the column in which is the text to be used in the inference in located
                              out_path:        <str>    Path to save the model's predictions.
                              batch_size:      <int>    Batch size.
                              state_dict_path: <str>    Path to the fine tuned model to be loaded.
                              labels:          <dict>   Dictionary of pairs values-labels
```
For examples of config files see `example_train.yml` and `example_predict.yml`
