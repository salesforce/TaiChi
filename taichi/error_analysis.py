import pandas as pd
import numpy as np

from transformers import AutoTokenizer

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
)

import os
import re

import matplotlib.pyplot as plt



class ErrorAnalysis(object):
    def __init__(self, save_dir="./error_analysis"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_pr_curve_plot(self, preds, test_labels, unique_labels, save_filename="multiclass-pr-curve.png"):

        Y_test = label_binarize(test_labels, classes=range(len(unique_labels)))
        y_score = preds[:,:,0]
        
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(len(unique_labels)):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            Y_test.ravel(), y_score.ravel()
        )
        average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot()
        _ = display.ax_.set_title("Micro-averaged over all classes")

        # setup plot details
        cmap = plt.cm.tab20
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        colors = cmaplist[:len(unique_labels)]

        _, ax = plt.subplots(figsize=(14, 16))

        f_scores = np.linspace(0.2, 1.0, num=5)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

        for i, color in zip(range(len(unique_labels)), colors):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax, name=f"Precision-recall for class: {unique_labels[i]}", color=color)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Extension of Precision-Recall curve to multi-class")

        plt.show()
        save_path = os.path.join(self.save_dir, save_filename)
        plt.savefig(save_path)


    def save_confusion_matrix_plot(self, preds, test_labels, unique_labels, save_filename="confusion_matrix.png"):
        cm = confusion_matrix(test_labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                      display_labels=unique_labels)
        disp.plot(xticks_rotation='vertical')
        plt.tight_layout()
        plt.show()
        save_path = os.path.join(self.save_dir, save_filename)
        plt.savefig(save_path)


    def get_intent_classification_report(self, preds, test_labels, unique_labels):
        if len(unique_labels) not in set(preds):
            report = classification_report(test_labels, preds, target_names=unique_labels, output_dict=True, zero_division=1)
        else:
            unique_labels += ["OOD"]
            report = classification_report(test_labels, preds, target_names=list(dict.fromkeys(unique_labels)), output_dict=True, zero_division=1)
        report_df = pd.DataFrame(report).transpose()
        return report_df


    def save_intent_classification_report(self, preds, test_labels, unique_labels, save_filename="intent_classification_report.csv"):
        report_df = self.get_intent_classification_report(preds, test_labels, unique_labels)
        save_path = os.path.join(self.save_dir, save_filename)  
        report_df.to_csv(save_path)


    def get_misclassified_instances(self, encoded_inputs, preds, test_labels, unique_labels, 
                                    multilingual_idx2label, language, tokenizer=None, train_labels=None):
        if tokenizer == None:
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # decode all examples to find misclassified examples
        decoded_inputs = []
        for i,j in enumerate(test_labels):
            decoded_input = tokenizer.decode(encoded_inputs[len(unique_labels) * i], skip_special_tokens=False)
            decoded_input = re.split("<s>|</s>", decoded_input)[1].strip()
            decoded_inputs.append(decoded_input)

        # detect misclassified examples
        misclassified = []
        for pred, test_label, utterance in zip(preds, test_labels, decoded_inputs):
            if not train_labels:
                if pred != test_label:
                    if pred != len(unique_labels):
                        misclassified.append((utterance, multilingual_idx2label[language][pred], multilingual_idx2label[language][test_label]))
                    else:
                        misclassified.append((utterance, "OOD", multilingual_idx2label[language][test_label]))
            else:
                if train_labels[pred] != test_label:
                    if train_labels[pred] != len(unique_labels):
                        misclassified.append((utterance, multilingual_idx2label[language][train_labels[pred]], multilingual_idx2label[language][test_label]))
                    else:
                        misclassified.append((utterance, "OOD", multilingual_idx2label[language][test_label]))                            

        misclassified_df = pd.DataFrame.from_records(misclassified, columns=['utterance', 'prediction', 'ground_truth'])
        return misclassified_df


    def save_misclassified_instances(self, encoded_inputs, preds, test_labels, unique_labels, 
                                    multilingual_idx2label, language, tokenizer=None, train_labels=None,
                                    save_filename='misclassified_examples.csv'):

        misclassified_df = self.get_misclassified_instances(encoded_inputs, preds, test_labels, unique_labels, 
                                                            multilingual_idx2label, language, tokenizer=None,
                                                            train_labels=train_labels)
        save_path = os.path.join(self.save_dir, save_filename)
        misclassified_df.to_csv(save_path)
