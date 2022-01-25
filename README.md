
# TaiChi

## Introduction
[Tai Chi](https://en.wikipedia.org/wiki/Tai_chi#) ☯️ , known as a Chinese martial art, emphasizes on practicing "smart strength" like the leverage of joints to gain great power with small efforts. This philiosophy interestingly fits perfectly into few-shot learning (FSL) research -- with "smart tricks", people try to train models with good performance using small amount of data. So we name our FSL library as Taichi in the hope that it will help your model training in low data scenario. 

Over last few years, we have seen great progress in FSL research thanks to the work in pre-training, meta-learning, data augmentation, and public benchmark datasets. Since data collection and labeling are often expensive and time-consuming, breakthroughs in FSL research have huge potential use cases in ML/DL industry. The Salesforce Research team has also done a lot of FSL related projects for research and application purposes, please feel free to check out our publications in FSL and other areas [here](https://www.salesforceairesearch.com/research). 

The Taichi library actually serves as an API hub for various effective methods proposed by the Salesforce Research team. We are currently releasing Taichi 1.0, which contains two main FSL methods: [DNNC](https://arxiv.org/abs/2010.13009) and [USLP](https://aclanthology.org/2021.nlp4convai-1.2/). These two methods are mainly for few-shot intent classification. We are working on including more useful FSL methods into Taichi, stay tuned for next release!

## 📋 Taichi 1.0 feature checklist

1. Pythonic API, “*from taichi import few_shot_learning_method”*
2. Based on pyTorch and Huggingface [transformers](https://github.com/huggingface/transformers) library
3. Included two recently published few-shot methods: [DNNC](https://arxiv.org/abs/2010.13009) and [USLP](https://aclanthology.org/2021.nlp4convai-1.2/)
4. Data sampling and error analysis API
5. Examples on [CLINC150](https://github.com/clinc/oos-eval/tree/master/data) dataset for quick start
6. Pre-trained English and multi-lingual transformer models and pre-processed CLINC150 dataset [here](https://console.cloud.google.com/storage/browser/sfr-few-shot-research)

## ⚙️ Methods: DNNC & USLP
The following figure provides a quick comparison of standard intent classification, DNNC, and USLP. In short, both DNNC and USLP are based upon NLI-style classification, DNNC reframes classification as entailment prediction between query and utterances in the training set while USLP tries to predict entailment relationship of utterance and semantic labels. Please refer to our [DNNC](https://arxiv.org/abs/2010.13009) and [USLP](https://aclanthology.org/2021.nlp4convai-1.2/) paper for more details.
![Algorithms](./readme/USLP_and_DNNC_description.png)

## 🚀 Models
We are also sharing the backbone models for DNNC and USLP. The models are based upon pubic pre-trained models from Huggingface and further tuned with NLI dataset to make them adapated to NLI-style classification.

1.  [nli-pretrained-roberta-base](https://console.cloud.google.com/storage/browser/sfr-few-shot-research/model/nli-pretrained-roberta-base), English only model
2.  [nli-pretrained-xlm-roberta-base](https://console.cloud.google.com/storage/browser/sfr-few-shot-research/model/nli-pretrained-xlm-roberta-base), based upon [XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlmroberta) model, which supports 100 languages, can be used for multi/cross-lingual projects

Please refer to the NLI pre-training pipeline [here](https://github.com/salesforce/DNNC-few-shot-intent) if you would like to pre-train a new model.

## 🛢 Data
We use [CLINC150 Dataset](https://github.com/clinc/oos-eval/tree/master/data) for benchmark and tutorials. The original `data_small.json` is sub-sampled and futher processed. User can download the processed dataset from [here](https://console.cloud.google.com/storage/browser/sfr-few-shot-research/data). 

## 🤔 Undersanding Taichi 1.0 API

**1. Data Sampling**

1. The following step imports the Data Pipeline object for quick data sampling
    1. `from taichi.data_pipeline import DataPipeline`
2. The following step sets up the data pipeline object with the dataset name, path and language

    1. `dp = DataPipeline(name=“clinc150”, data_path=“full path to data file in csv or json, edit accordingly”)`
    2. Expects json data file in the following format:
        1. `{split: list(list containing utterance and label)}`
            - Example: `{'train':[[utterance1, label1], [utterance2, label2], ... 'test':[[...]]}`
        2. The data format is as found in CLINC150 dataset
    3. Expects csv data file in the following format:
        1. `utterance, label (no headers and no index)`
            - Example: `book a ticket from San Francisco to New York, Book a Flight`
1. Based on the data file and format received (csv/json), we can subsample the input data file and save it as csv or json in the path (`save_dir`) of our choice
    1. to save to csv, use the following command:
        1. `dp.save_subsampled_data_to_csv(save_dir="./data/CLINC150/1-shot", split='train', n_shot=1, is_json=True, random_state=42,  save_filename="train.csv")`
            - Here, the default split `train`  (will check for right split name and throw exception in case of incorrect split name; also does not matter if the data source is `csv`) in the `CLINC150` dataset json file (`is_json=True`, False in case of data source being a `csv` ) gets subsampled into `10` samples per class (will check if possible_ and saved in `os.path.join(save_dir, save_filename)`  creating the path if it doesn’t exist in the process as `csv` file in the format mentioned above in 2c
    2. we can save our file as json in much the same way with the following command:
        1. `dp.save_subsampled_data_to_json(save_dir="./data/CLINC150/1-shot", split='train', n_shot=1, is_json=True, random_state=42, orient='records', save_filename="1-shot-train.json")`

**2. Modifying Config Parameters**

1. We have individual config files containing hyperparameters for USLP and DNNC models. Please find below an example of the config file for USLP (the DNNC config file also has the same parameters):
    ```python
    {
        "model": "roberta-base",
        "checkpoint_dir": "./model/nli-pretrained-roberta-base/uslp",
        "train_data_path": "./data/CLINC150/5-shot/train.csv",
        "test_data_path": "./data/CLINC150/5-shot/test.csv",
        "ood_train_data_path": "./data/CLINC150/5-shot/ood_train.csv",
        "ood_test_data_path": "./data/CLINC150/5-shot/ood_test.csv",
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-05,
        "no_cuda": false,
        "num_train_epochs": 200,
        "pretrained_model_path": "./model/nli-pretrained-roberta-base",
        "save_result_fp": "./data/CLINC150/5-shot/uslp_inference.json",
        "seed": 42,
        "max_seq_length": 64,
        "test_batch_size": 128,
        "train_batch_size": 128,
        "transform_labels": false,
        "warmup_proportion": 0.1,
        "weight_decay": 0.0001,
        "threshold": 0.01
    }
    ```
    - Let us dive deeper into some of the individual parameters and groups of parameters to understand why they are needed
        1. `model` defines the model name, e.g. roberta-base, TaiChi will use this information to load pretrained tokenizer from huggingface;
        2. `checkpoint_dir` is the user defined directory for saving models after finetuning;
        3. `train_data_path`, `test_data_path`, `ood_train_data_path` and `ood_test_data_path` are user defined paths for the model to know where to take the data from;
        4. `pretrained_model_path` specifies the path to the model pretrained on general NLI datasets;
        5. `save_result_fp` is the path to store the inference results in terms of threshold, in-domain accuracy, precision, recall, and f1 macro along with ood-recall in a `json` format
        6. Other configuration parameters are mostly about hyperparameters for training.

**3. Run Code End-to-End**

- Please find a quick snapshot on how the USLP model can be trained as below
    ```python
    from taichi import uslp # import algorithm
    
    uslp_model = uslp.USLP() # instantiate algorithm (default config path set to ./taichi/uslp_config.json)
    
    uslp_model.init() # initialize the data and model
    
    uslp_model.train() # model training
    
    uslp_model.eval() # model evaluation
    ```

**Results From Paper (*Focus on DNNC and USLP-T*)**
![paper-results](./readme/USLP_Paper_Results.png)

**Benchmark results on CLINC150**
- Computing environment: torch==1.7.1, transformers==4.5.1, A100 GPU (user might expect results to vary with different software versions/hardwares)
- Hyper-parameter
  - threshold: 0.01
  - training batch size: 128
  - epochs: 200
  - learning rate: 5e-5

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
3. The tricks we implemented are NOT part of the DNNC code we share since TaiChi is designed for few shot learning use case.


**Testing**

To test if the models work as expected, please run `test_uslp.py` and `test_dnnc.py` which can be found in the `tests` directory.
Please note that the config files (`test_uslp_config.json` and `test_dnnc_config.json`) would have to be altered accordingly to point to the model and data we use to evaluate the tests. For USLP, we run 1-shot experiment on CLINC150 and for DNNC, we run 5-shot experiment on CLINC150.


## References

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
3. [XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)
4. [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)
5. [USLP: Few-Shot Intent Classification by Gauging Entailment Relationship Between Utterance and Semantic Label](https://aclanthology.org/2021.nlp4convai-1.2/)
6. [DNNC: Discriminative Nearest Neighbor Few-Shot Intent Detection by Transferring Natural Language Inference](https://arxiv.org/abs/2010.13009) 
7. [CLINC150 Dataset](https://github.com/clinc/oos-eval/tree/master/data)


## Contact
Please feel free to reach out to jqu@salesforce.com for questions or feedback.
