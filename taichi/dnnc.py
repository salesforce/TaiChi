import os
import sys
import csv
import json
import logging
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Iterable, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModelForSequenceClassification,
)


from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
)

import copy

from taichi.config import Config
from taichi.error_analysis import ErrorAnalysis

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ENTAILMENT = 0
NON_ENTAILMENT = 1

THRESHOLD_MIN = 0
THRESHOLD_MAX = 1.0
THRESHOLD_STEP = 0.01


class DNNC(object):
    """
    Main class for DNNC including initialization, training and evaluation
    """

    def __init__(self, config_file="./taichi/dnnc_config.json"):
        with open(config_file, "r") as f:
            config_dict = json.loads(f.read())
        self.config = Config(config_dict)

        self.device = None
        self.tokenizer = None
        self.model = None
        # train parameters
        self.train_data = None
        self.train_labels = None
        self.pos_train_dataloader = None
        self.neg_train_dataloader = None
        self.ood_train_dataloader = None
        # test parameters
        self.test_data = None
        self.unique_test_labels = None
        self.test_labels = None
        self.test_label_ids = None
        self.ood_test_data = None

    def init(self):
        """
        initialize and setup environment, data and model for training
        """
        config = self.config
        logger.info("config: {}".format(config))

        # set up device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
        )

        logger.info("device: {}".format(self.device))

        # set up seeds
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model)

        # train_dataloader
        aggregated_data_df = pd.read_csv(
            config.train_data_path, names=["utterance", "language", "label"]
        )
        self.train_data = list(aggregated_data_df.utterance)
        self.train_labels = list(aggregated_data_df.label)
        self.train_languages = list(aggregated_data_df.language)

        unique_languages = sorted(list(set(self.train_languages)))

        # map languages to unique labels it contains examples of in the data
        lang2label = {}
        for language in unique_languages:
            lang_df = aggregated_data_df.loc[aggregated_data_df.language == language]
            lang2label[language] = sorted(list(set(lang_df.label)))

        multilingual_label2idx = {}
        self.multilingual_idx2label = {}
        for language, labels in lang2label.items():
            if language not in multilingual_label2idx:
                multilingual_label2idx[language] = {}
                self.multilingual_idx2label[language] = {}
                for index, label in enumerate(labels):
                    multilingual_label2idx[language][label] = index
                    self.multilingual_idx2label[language][index] = label

        self.train_label_ids = [
            multilingual_label2idx[lang][lbl]
            for lbl, lang in zip(self.train_labels, self.train_languages)
        ]
        # get unique labels and make sure the order stays consistent by sorting
        unique_train_labels = sorted(list(set(self.train_labels)))

        # create positive and negative train examples - assumes the language throughout is same
        positive_train_examples = []
        negative_train_examples = []
        for i in range(len(self.train_data)):
            for j in range(len(self.train_data)):
                if self.train_labels[i] == self.train_labels[j]:
                    if j <= i:
                        continue

                    positive_train_examples.append(
                        (
                            self.train_data[i],
                            self.train_data[j],
                            self.train_languages[j],
                        )
                    )

                    positive_train_examples.append(
                        (
                            self.train_data[j],
                            self.train_data[i],
                            self.train_languages[i],
                        )
                    )

                else:
                    negative_train_examples.append(
                        (
                            self.train_data[i],
                            self.train_data[j],
                            self.train_languages[i],
                        )
                    )

        # modify positive and negative examples to remove the language element
        positive_train_examples = [(d[0], d[1]) for d in positive_train_examples]
        negative_train_examples = [(d[0], d[1]) for d in negative_train_examples]

        positive_train_features = self.tokenizer(
            positive_train_examples,
            return_tensors="pt",
            padding="max_length",
            max_length=config.max_seq_length,
            truncation=True,
        )
        num_positive_train_examples = len(positive_train_examples)

        del positive_train_examples # free memory

        # checking if token_type_ids available for positive examples - should be same for all
        pos_token_type_ids = positive_train_features.get("token_type_ids", None)
        if pos_token_type_ids is None:
            self.is_bert_type_tokenizer = False
        else:
            self.is_bert_type_tokenizer = True

        negative_train_features = self.tokenizer(
            negative_train_examples,
            return_tensors="pt",
            padding="max_length",
            max_length=config.max_seq_length,
            truncation=True,
        )
        num_negative_train_examples = len(negative_train_examples)

        del negative_train_examples # free memory

        positive_train_labels = torch.tensor(
            [ENTAILMENT for _ in range(num_positive_train_examples)]
        )

        if self.is_bert_type_tokenizer != True:
            positive_train_dataset = TensorDataset(
                positive_train_features["input_ids"],
                positive_train_features["attention_mask"],
                positive_train_labels,
            )
        else:
            positive_train_dataset = TensorDataset(
                positive_train_features["input_ids"],
                positive_train_features["attention_mask"],
                positive_train_features["token_type_ids"],
                positive_train_labels,
            )

        negative_train_labels = torch.tensor(
            [NON_ENTAILMENT for _ in range(num_negative_train_examples)]
        )

        if self.is_bert_type_tokenizer != True:
            negative_train_dataset = TensorDataset(
                negative_train_features["input_ids"],
                negative_train_features["attention_mask"],
                negative_train_labels,
            )
        else:
            negative_train_dataset = TensorDataset(
                negative_train_features["input_ids"],
                negative_train_features["attention_mask"],
                negative_train_features["token_type_ids"],
                negative_train_labels,
            )

        self.pos_train_dataloader = DataLoader(
            positive_train_dataset,
            batch_size=int(config.train_batch_size // 2),
            shuffle=True,
        )

        self.neg_train_dataloader = DataLoader(
            negative_train_dataset,
            batch_size=config.train_batch_size // 4,
            shuffle=True,
        )

        # ood_train_dataloader
        ood_train_data = []
        with open(config.ood_train_data_path) as file:
            csv_file = csv.reader(file)
            for line in csv_file:
                ood_train_data.append(line[0])
        ood_train_examples = []

        # for dnnc, unique train labels should get replaced by train data
        for e in ood_train_data:
            for l in self.train_data:
                ood_train_examples.append((e, l))
        ood_train_features = self.tokenizer(
            ood_train_examples,
            return_tensors="pt",
            padding="max_length",
            max_length=config.max_seq_length,
            truncation=True,
        )
        num_ood_train_examples = len(ood_train_examples)

        del ood_train_examples # free memory

        ood_train_labels = torch.tensor([NON_ENTAILMENT for _ in range(num_ood_train_examples)])
        if self.is_bert_type_tokenizer != True:
            ood_train_dataset = TensorDataset(
                ood_train_features["input_ids"],
                ood_train_features["attention_mask"],
                ood_train_labels,
            )
        else:
            ood_train_dataset = TensorDataset(
                ood_train_features["input_ids"],
                ood_train_features["attention_mask"],
                ood_train_features["token_type_ids"],
                ood_train_labels,
            )
        self.ood_train_dataloader = DataLoader(
            ood_train_dataset, batch_size=config.train_batch_size // 4, shuffle=True
        )

        # load test dataloader
        self.test_data, self.test_labels, self.test_languages = [], [], []
        with open(config.test_data_path) as file:
            csv_file = csv.reader(file)
            for line in csv_file:
                self.test_data.append(line[0])
                self.test_languages.append(line[1])
                self.test_labels.append(line[2])

        # detect language based on assumption that test data would have only one language

        self.unique_test_labels = lang2label
        self.test_label_ids = [
            multilingual_label2idx[lang][lbl]
            for lbl, lang in zip(self.test_labels, self.test_languages)
        ]

        # load ood test dataloader
        self.ood_test_data = []
        with open(config.ood_test_data_path) as file:
            csv_file = csv.reader(file)
            for line in csv_file:
                self.ood_test_data.append(line[0])

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.pretrained_model_path
        )

        if config.error_analysis:
            self.ea = ErrorAnalysis(config.error_analysis_dir)

    def train(self):
        # load pretrained NLI model
        config = self.config
        model = self.model
        if config.freeze_embedding:
            for p in model.roberta.embeddings.parameters():
                p.requires_grad = False
        model.to(self.device)

        num_train_optimization_steps = (
            len(self.pos_train_dataloader) * config.num_train_epochs
        )

        # load optimizer and scheduler
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                    if p.requires_grad
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay if p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=config.learning_rate, eps=1e-8
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                num_train_optimization_steps * config.warmup_proportion
            ),
            num_training_steps=num_train_optimization_steps,
        )

        # loss function
        loss_fct = nn.CrossEntropyLoss()

        # training pipeline
        logging.info("***** Running training *****")
        logging.info("  Num positive examples = {}".format(len(self.train_data)))
        logging.info("  Batch size = %d", config.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)

        progress_bar = tqdm(
            total=num_train_optimization_steps, dynamic_ncols=True, initial=0
        )
        for epoch in range(config.num_train_epochs):
            for _, pos_batch in enumerate(self.pos_train_dataloader):
                model.train()
                neg_batch = next(iter(self.neg_train_dataloader))
                ood_batch = next(iter(self.ood_train_dataloader))
                if self.is_bert_type_tokenizer != True:
                    pos_input_ids, pos_attention_mask, pos_labels = pos_batch
                    neg_input_ids, neg_attention_mask, neg_labels = neg_batch
                    ood_input_ids, ood_attention_mask, ood_labels = ood_batch
                else:
                    (
                        pos_input_ids,
                        pos_attention_mask,
                        pos_token_type_ids,
                        pos_labels,
                    ) = pos_batch
                    (
                        neg_input_ids,
                        neg_attention_mask,
                        neg_token_type_ids,
                        neg_labels,
                    ) = neg_batch
                    (
                        ood_input_ids,
                        ood_attention_mask,
                        ood_token_type_ids,
                        ood_labels,
                    ) = ood_batch
                input_ids = torch.cat((pos_input_ids, neg_input_ids, ood_input_ids), 0)
                attention_mask = torch.cat(
                    (pos_attention_mask, neg_attention_mask, ood_attention_mask), 0
                )
                if self.is_bert_type_tokenizer != True:
                    token_type_ids = None
                else:
                    token_type_ids = torch.cat(
                        (pos_token_type_ids, neg_token_type_ids, ood_token_type_ids), 0
                    )
                labels = torch.cat((pos_labels, neg_labels, ood_labels), 0)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                if self.is_bert_type_tokenizer == True:
                    token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )[0]
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                preds = logits.cpu().detach().numpy()
                preds = np.argmax(preds, axis=1)
                truth = labels.cpu().detach().numpy().tolist()
                acc = accuracy_score(truth, preds)
                p = precision_score(truth, preds, average="macro", zero_division=1)
                r = recall_score(truth, preds, average="macro", zero_division=1)
                progress_bar.set_description(
                    "Epoch "
                    + str(epoch)
                    + " ### loss = %g ### acc = %g ### p = %g ### r = %g ###"
                    % (loss.item(), acc, p, r)
                )
                progress_bar.update(1)

            # save model when finish
            if config.checkpoint_dir and epoch == config.num_train_epochs - 1:
                if not os.path.isdir(config.checkpoint_dir):
                    os.mkdir(config.checkpoint_dir)
                model.save_pretrained(config.checkpoint_dir)

        progress_bar.close()

    def eval(self):

        config = self.config
        model = AutoModelForSequenceClassification.from_pretrained(
            config.saved_model_path
        )
        model.to(self.device)
        language = self.test_languages[0]
        unique_labels = self.unique_test_labels[language]
        if config.transform_labels:
            unique_labels = [
                str(multilingual_label2idx[language][l])
                for l in self.unique_test_labels[language]
            ]
        res_indomain, prob_indomain = self._evaluation_indomain(
            model,
            language,
            self.test_data,
            self.test_label_ids,
            self.tokenizer,
            self.train_data,
            self.train_label_ids,
            unique_labels,
            self.device,
        )
        # compute index to print per threshold entered
        threshold_index = int(config.threshold * 100)
        logger.info(
            f"in-domain eval at {config.threshold} threshold: {res_indomain[threshold_index]}"
        )
        res_ood_recall, prob_ood = self._evaluation_ood_recall(
            model,
            language,
            self.ood_test_data,
            self.tokenizer,
            self.train_data,
            unique_labels,
            self.device,
        )

        res_ood_prec_f1 = self._evaluation_ood_precision_f1(prob_indomain, prob_ood)

        res_ood_precision = [res[1] for res in res_ood_prec_f1]  # get precision
        res_ood_f1 = [res[2] for res in res_ood_prec_f1]  # get F1
        res_ood = [
            (recall[0], recall[1], precision, f1)
            for recall, precision, f1 in zip(
                res_ood_recall, res_ood_precision, res_ood_f1
            )
        ]

        logger.info(
            f"ood eval at {config.threshold} threshold: {res_ood[threshold_index]}"
        )
        logger.info("***" * 6)

        # save final results
        if config.save_result_fp is not None:
            res2save = {}
            config_dict = config.__dict__.copy()
            del config_dict["_keys"]
            res2save["config"] = config_dict
            res2save["test-indomain"] = res_indomain
            res2save["test-ood"] = res_ood

            if os.path.isfile(config.save_result_fp):
                with open(config.save_result_fp, "r") as f:
                    final_res = json.load(f)
                    final_res["all_res"].append(res2save)
            else:
                final_res = {"all_res": [res2save]}

            with open(config.save_result_fp, "w") as f:
                json.dump(final_res, f, indent=4)

    def _evaluation_indomain(
        self,
        model,
        language,
        test_data,
        test_labels,
        tokenizer,
        train_data,
        train_labels,
        unique_labels,
        device,
        eval_batch_size=128,
    ):
        model.eval()

        test_data_in_nli_format = []
        test_labels_in_nli_format = []
        for i, sample1 in enumerate(test_data):
            for j, sample2 in enumerate(train_data):
                test_data_in_nli_format.append((sample1, sample2))

        features = tokenizer(
            test_data_in_nli_format,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.max_seq_length,
            truncation=True,
        )

        if self.is_bert_type_tokenizer != True:
            dataset = TensorDataset(features["input_ids"], features["attention_mask"])
        else:
            dataset = TensorDataset(
                features["input_ids"],
                features["attention_mask"],
                features["token_type_ids"],
            )
        dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

        preds = None
        encoded_inputs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if self.is_bert_type_tokenizer != True:
                    input_ids, attention_mask = batch
                    token_type_ids = None
                else:
                    input_ids, attention_mask, token_type_ids = batch
                if self.config.error_analysis:
                    for i, input_id in enumerate(input_ids):
                        encoded_inputs.append(input_id)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                if self.is_bert_type_tokenizer == True:
                    token_type_ids = token_type_ids.to(device)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )[0]
                pred = nn.Softmax(dim=-1)(logits).cpu().detach().numpy()
                if preds is None:
                    preds = pred
                else:
                    preds = np.concatenate((preds, pred))
        preds = np.reshape(preds, (-1, len(train_data), 2))
        max_pos_idx = np.argmax(preds[:, :, 0], axis=1)
        max_prob = np.max(preds[:, :, 0], axis=1)

        res = []
        for threshold in np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP):
            preds = []
            for prob, pred_label in zip(max_prob, max_pos_idx):
                if prob > threshold:
                    preds.append(train_labels[pred_label])
                else:
                    preds.append(len(unique_labels))

            if threshold == self.config.threshold:
                if self.config.error_analysis:
                    # default save path used here, set own path by assigning custom save_path argument
                    self.ea.save_misclassified_instances(
                        encoded_inputs,
                        preds,
                        test_labels,
                        unique_labels,
                        self.multilingual_idx2label,
                        language,
                        tokenizer=tokenizer,
                    )
                    self.ea.save_intent_classification_report(
                        preds, test_labels, unique_labels
                    )
                    self.ea.save_confusion_matrix_plot(
                        preds, test_labels, unique_labels
                    )

            acc = accuracy_score(test_labels, preds)
            prec = precision_score(test_labels, preds, average="macro", zero_division=1)
            recall = recall_score(test_labels, preds, average="macro", zero_division=1)
            f1 = f1_score(test_labels, preds, average="macro", zero_division=1)
            res.append((threshold, acc, prec, recall, f1))
        return res, max_prob

    def _evaluation_ood_recall(
        self,
        model,
        language,
        ood_test_data,
        tokenizer,
        train_data,
        unique_labels,
        device,
        eval_batch_size=128,
    ):

        test_labels = [len(unique_labels) for _ in ood_test_data]
        model.eval()

        test_data_in_nli_format = []
        for e in ood_test_data:
            for l in train_data:
                test_data_in_nli_format.append((e, l))

        features = tokenizer(
            test_data_in_nli_format,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.max_seq_length,
            truncation=True,
        )

        if self.is_bert_type_tokenizer != True:
            dataset = TensorDataset(features["input_ids"], features["attention_mask"])
        else:
            dataset = TensorDataset(
                features["input_ids"],
                features["attention_mask"],
                features["token_type_ids"],
            )

        dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

        preds = None
        encoded_inputs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if self.is_bert_type_tokenizer != True:
                    input_ids, attention_mask = batch
                    token_type_ids = None
                else:
                    input_ids, attention_mask, token_type_ids = batch
                if self.config.error_analysis:
                    for i, input_id in enumerate(input_ids):
                        encoded_inputs.append(input_id)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                if self.is_bert_type_tokenizer == True:
                    token_type_ids = token_type_ids.to(device)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )[0]
                pred = nn.Softmax(dim=-1)(logits).cpu().detach().numpy()
                if preds is None:
                    preds = pred
                else:
                    preds = np.concatenate((preds, pred))

        preds = np.reshape(preds, (-1, len(train_data), 2))
        max_pos_idx = np.argmax(preds[:, :, 0], axis=1)
        max_prob = np.max(preds[:, :, 0], axis=1)

        res = []
        for threshold in np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP):
            preds = []
            ood_preds = []
            ood_gt = []
            for prob, pred_label in zip(max_prob, max_pos_idx):
                ood_gt.append(NON_ENTAILMENT)
                if prob > threshold:
                    preds.append(pred_label)
                    ood_preds.append(ENTAILMENT)
                else:
                    preds.append(len(unique_labels))
                    ood_preds.append(NON_ENTAILMENT)

            if threshold == self.config.threshold:

                ood_labels = ["NOT OOD", "OOD"]
                if self.config.error_analysis:
                    self.ea.save_intent_classification_report(
                        ood_preds, ood_gt, ood_labels, save_filename="ood_report.csv"
                    )
                    self.ea.save_confusion_matrix_plot(
                        ood_preds,
                        ood_gt,
                        ood_labels,
                        save_filename="ood_confusion_matrix.png",
                    )

            recall = recall_score(ood_gt, ood_preds, zero_division=1)
            res.append((threshold, recall))
        return res, max_prob

    def _evaluation_ood_precision_f1(self, in_domain_probs, ood_probs):
        labels = [ENTAILMENT for _ in in_domain_probs] + [
            NON_ENTAILMENT for _ in ood_probs
        ]
        max_conf = np.concatenate((in_domain_probs, ood_probs))
        res = []
        for threshold in np.arange(0, 0.91, 0.01):
            preds = []
            for prob in max_conf:
                if prob > threshold:
                    preds.append(ENTAILMENT)
                else:
                    preds.append(NON_ENTAILMENT)

            prec = precision_score(labels, preds, zero_division=1)
            f1 = f1_score(labels, preds, zero_division=1)
            res.append((threshold, prec, f1))
        return res
