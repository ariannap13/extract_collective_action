This folder contains all the code and data to reproduce the analyses of the paper "Extracting Participation in Collective Action from Social Media". 

All analysis are performed with Python 3.8.16 and the list of required packages versions can be found in ``requirements.txt``.

Here is a brief description of the sub-folders, further details can be found in the `README.md` file placed in the folders containing code (i.e. all but `./data`, `./models` and `./downstream` - which is self-contained since it only involves a notebook for downstream analyses):
* `./data/` contains the data files used for the analysis. When not present in the folder (e.g. stance detection dataset), the datasets are either taken from other papers with references in the main paper or defined as working products. While these latter are not shared, authors can be contacted for access.
* `./data_augmentation/` contains the scripts to perform data augmentation by using synthetic generation and extension through Reddit structural characteristics.
* `./preprocessing/` contains some useful preprocessing and sampling scripts, including the script to extract action-focused sentences from text.
* `./baselines/` contains the scripts to build the classifiers used as a baseline comparison.
* `./bert_classifier/` contains the scripts to fine-tune the RoBERTa classifier for both the binary and the multi-class tasks and run the corresponding inference.
* `./llm_zeroshot/` contains the code for the application of LLaMa3 as a zero-shot classifier in the context of both the binary and the multi-class tasks and run the corresponding inference.
* `./llm_finetune/` contains the code to finetune LLaMa3 with SFT and DPO approaches and run the corresponding inference.
* `./models/` contains the trained models. Due to size constraints, models will be shared on HuggingFace upon acceptance.
* `./validation/` contains the code to perform the comparison of the approach with topic modeling and stance detection and to evaluate the impact of the dictionary of collective action words on the results. It also contains the analysis of UK parliamentary debates data.
* `./downstream/`contains the code to perform the downstream tasks (Reddit characterization and climate change case study) and obtain the corresponding plots.