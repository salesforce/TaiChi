# TaiChi

### Introduction
[Tai Chi](https://en.wikipedia.org/wiki/Tai_chi#) ☯️ , known as a Chinese martial art, emphasizes on practicing "smart strength" like the leverage of joints to gain great power with small efforts. This philiosophy interestingly fits perfectly into few-shot learning (FSL) research -- with "smart tricks", people try to train models with good performance using small amount of data. So we name our FSL library as Taichi in the hope that it will help your model training 

Over last few years, we have seen great progress in FSL

**Enter Taichi - An open source Python library for few shot learning**

1. Modular and extensible API design, “*from taichi import few_shot_learning_method”*
2. Two few shot methods have been implemented, USLP + DNNC
3. Supports quick data sampling and error analysis
4. Active Development GitHub repo: [https://github.com/MetaMind/taichi-internal](https://github.com/MetaMind/taichi-internal)
5. The data and pretrained nli models can be found in this [GCS bucket](https://console.cloud.google.com/storage/browser/sfr-few-shot-research)

### High Level Details on Algorithms

![Algorithms](./readme/USLP_and_DNNC_description.png)
1. As explained above, the difference between USLP and DNNC is that USLP reframes the classification task as an entailment prediction between query utterance and semantic label, while DNNC reframes the task as an entailment prediction between query and example utterances
2. The original models used for nli-pretraining were [RoBERTa](https://arxiv.org/abs/1907.11692), [XLM-Roberta](https://arxiv.org/abs/1911.02116) and [MiniLM](https://arxiv.org/abs/2002.10957) distilled from XLM-RoBERTa
3. Following table briefly describes their individual characteristics
    |Characteristics    |RoBERTa |XLM-RoBERTa   |MiniLM|
    |---    |---    |---    |---    |
    |Description   |Baseline model  |This model supports multiple languages and can be considered a multilingual version of RoBERTa |This model is a leaner model with XLM-RoBERTa as teacher and is compressed with deep self-attention distillation|
    |Strength   |Strong performance on English datasets |Strong performance on multilingual datasets    |Competitive performance on multilingual datasets and low memory footprint  |
    |Weakness   |No support for multiple languages  |Bulky model    |Not as good as XLM-RoBERTa in performance  |
    1. The pretraining of the above NLI models is covered [here](https://github.com/salesforce/DNNC-few-shot-intent)
4. The USLP model is typically faster than DNNC in training as number of labels are usually lower than the number of examples
 

### High Level Details on Data
1. The original CLINC150 data file (`data_small.json`) used was in json format and was therefore preprocessed into corresponding csv with no headers and index
2. We utilized the data sampling code to get our splits for training, testing and OOD from the json files into respective csv files
3. We further processed the OOD files to remove the label column

### How to Run the Code

**Data Sampling**

1. The following step imports the Data Pipeline object for quick data sampling
    1. `from taichi.data_pipeline import DataPipeline`
2. The following step sets up the data pipeline object with the dataset name, path and language

    1. `dp = DataPipeline(name=“clinc150”, data_path=“full path to data file in csv or json, edit accordingly”, language=“en_US by default”)`
    2. Expects json data file in the following format:
        1. `{split: list(list containing utterance and label)}`
            1. Example: `{'train':[[utterance1, label1], [utterance2, label2], ... 'test':[[...]]}`
        2. The data format is as found in CLINC150 dataset
    3. Expects csv data file in the following format:
        1. `utterance,language,label (no headers and no index)`
            1. Example: `book a ticket from San Francisco to New York,en_US,Book a Flight`
1. Based on the data file and format received (csv/json), we can subsample the input data file and save it as csv or json in the path (`save_dir`) of our choice
    1. to save to csv, use the following command:
        1. `dp.save_subsampled_data_to_csv(save_dir="./data/CLINC150/1-shot", split='train', n_shot=1, is_json=True, random_state=42,  save_filename="train.csv")`
            1. Here, the default split `train`  (will check for right split name and throw exception in case of incorrect split name; also does not matter if the data source is `csv`) in the `CLINC150` dataset json file (`is_json=True`, False in case of data source being a `csv` ) gets subsampled into `10` samples per class (will check if possible_ and saved in `os.path.join(save_dir, save_filename)`  creating the path if it doesn’t exist in the process as `csv` file in the format mentioned above in 2c
    2. we can save our file as json in much the same way with the following command:
        1. `dp.save_subsampled_data_to_json(save_dir="./data/CLINC150/1-shot", split='train', n_shot=1, is_json=True, random_state=42, orient='records', save_filename="1-shot-train.json")`

**Modifying Config Parameters**

1. We have individual config files containing hyperparameters for USLP and DNNC models. Please find below an example of the config file for USLP (the DNNC config file also has the same parameters):
    ![Configuration](./readme/config.png)
    1. Let us dive deeper into some of the individual parameters and groups of parameters to understand why they are needed
        1. `bert-model` is for the pretrained tokenizer that we will be using to derive features from sequences
        2. `checkpoint_dir` is to save the model after training the model reformulating the classification task as an entailment prediction problem. We use `saved_model_path` for loading the trained model during evaluation. Therefore `checkpoint_dir` and `saved_model_path` are typically the same
        3. `train_data_path`, `test_data_path`, `ood_train_data_path` and `ood_test_data_path` are user defined paths for the model to know where to take the data from. One can see how the `save_dir` during the data sampling is used as input paths for our downstream classification task. One can adjust the paths per one’s convenience
        4. `pretrained_model_path` is self explanatory as we mention the pretrained model path we intend to use to reformulate and train again as an entailment prediction problem
        5. `save_result_fp` is the path to store the inference results in terms of threshold, in-domain accuracy, precision, recall, and f1 macro along with ood-recall in a `json` format
        6. `error_analysis` is a flag to indicate if the user is interested in some quick error analysis where on evaluation, the user can quickly gather where the model made mistakes (misclassifications), see a more detailed classification performance per class (classification report) and see visualizations in the form of confusion matrix for both in-domain and OOD samples. All these files get generated and saved in `error_analysis_dir`
        7. All other hyperparameters mentioned in the config are self explanatory and tweaking them can potentially help users in better performance on their datasets of choice

**Note:** The paths for the model and data (parameters) are currently generically set and will have to be modified by the user suitably in the config file to ensure smooth running of the library.


**Run Code End-to-End**

1. Please find a quick snapshot on how the USLP model can be trained as below
    ```python
    from taichi import uslp # import algorithm
    
    uslp_model = uslp.USLP() # instantiate algorithm (default config path set to ./taichi/uslp_config.json)
    
    uslp_model.init() # initialize the data and model
    
    uslp_model.train() # model training
    
    uslp_model.eval() # model evaluation
    ```

1. DNNC has the same easy API signature for training and evaluation
    ```python
    from taichi import dnnc # import algorithm
    
    dnnc_model = dnnc.DNNC() # instantiate algorithm (default config path set to ./taichi/dnnc_config.json)
    
    dnnc_model.init() # initialize the data and model
    
    dnnc_model.train() # model training
    
    dnnc_model.eval() # model evaluation
    ```

**Environment, Data, Model and Hyperparameters Summary**

1. Used torch==1.7.1 AND transformers==4.5.1 **(same as paper)** versions for the experiments on an a100 GPU
2. Used `data_small.json` from [CLINC150](https://github.com/clinc/oos-eval/tree/master/data) for benchmarking
3. Model: `roberta-base` implementation from Huggingface
4. Threshold: 0.01
5. Train Batch Size: 128
6. Num Train Epochs: 200
7. Learning Rate: 5e-5
8. No Data Augmentation

**Results From Paper (*Focus on DNNC and USLP-T*)**
![paper-results](./readme/USLP_Paper_Results.png)

**Results using Taichi Library**

Comparable results for USLP using Taichi to the results presented in the paper (USLP-T) for in-domain F1, OOD-Recall and OOD-Precision. Higher results for DNNC in comparison to results in the paper (DNNC) for in-domain F1 and OOD-Recall with comparable OOD-Precision.

|model	|samples per class	|in-domain F1	|OOD-recall	|OOD-precision	|
|---	|---	|---	|---	|---	|
|USLP	|full	|0.9459	|0.637	|0.947	|
|   |10	|0.892	|0.734	|0.854	|
|   |5	|0.8354	|0.679	|0.857	|
|   |1	|0.6667	|0.629	|0.664	|
|DNNC   |full	|0.9489	|0.25	|0.996	|
|   |10	|0.9203	|0.603	|0.933	|
|   |5	|0.902	|0.789	|0.858	|
|   |1	|NA	|NA	|NA	|

We also compare this with using off-the-shelf (not NLI-pretrained) BERT model (`bert-base-uncased`) and get the following results:

|model	|samples per class	|in-domain F1	|OOD-recall	|OOD-precision	|
|---	|---	|---	|---	|---	|
|USLP	|full	|0.9446	|0.722	|0.914	|
|   |10	|0.8838	|0.738	|0.836	|
|   |5	|0.8289	|0.772	|0.721	|
|   |1	|0.6526	|0.66	|0.584	|
|DNNC	|full	|0.9258	|0.329	|0.968	|
|   |10	|0.9055	|0.58	|0.898	|
|   |5	|0.8732	|0.737	|0.791	|
|   |1	|NA	|NA	|NA	|

**Notes on Full-Shot DNNC Experiments**

1. We faced OOM issues on running the DNNC code as is for these experiments. We tried the following as workaround:
   1. We reduced the number of negative nli pairs by random subsampling, using a ratio of negative to positive pairs (50 for our experiments) as a variable
   2. We processed the data in batches during training and inference
2. We ran these experiments for 10 epochs and it took ~35 hours to train on an A100 GPU for both `roberta-base` and `bert-base-uncased` models
   1. The OOD-recall results are *worse (lower)* most likely due to running these experiments on reduced number of epochs (10 as opposed to 200 for other experiments)
   2. The training time naturally blows up due to the algorithm design of generating negative and positive nli pairs
        1. If we consider CLINC150 full-shot experiment, the training data has 50 (`m`) examples per class and 150 classes (`n`) = 7500 examples `(m * n)`
        2. If we consider one example out of them and pair them to get positive and negative NLI pairs based on whether they belong to the same class, we get `(m-1)` 49 positive pairs and `(m * n - m)` 7450 negative pairs. The ratio between them `(m * n - m)/(m - 1)` is approximately equal to `n` which is 150 (152.04 in this case)
        3. If all pairs add up, the sheer number of examples makes it prohibitive to train the model and get results quickly.
3. The tricks we implemented are NOT part of the DNNC code we share, which makes it important to emphasize that it may face memory and training time issues if used on a large dataset.



**Testing**

To test if the models work as expected, please run `test_uslp.py` and `test_dnnc.py` which can be found in the `tests` directory.
Please note that the config files (`test_uslp_config.json` and `test_dnnc_config.json`) would have to be altered accordingly to point to the model and data we use to evaluate the tests. For USLP, we run 1-shot experiment on CLINC150 and for DNNC, we run 5-shot experiment on CLINC150.


### References

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
3. [XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)
4. [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)
5. [USLP: Few-Shot Intent Classification by Gauging Entailment Relationship Between Utterance and Semantic Label](https://aclanthology.org/2021.nlp4convai-1.2/)
6. [DNNC: Discriminative Nearest Neighbor Few-Shot Intent Detection by Transferring Natural Language Inference](https://arxiv.org/abs/2010.13009) 
7. [CLINC150 Dataset](https://github.com/clinc/oos-eval/tree/master/data)
