# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset
from tqdm import tqdm, trange
from transformers import (
    BertForSequenceClassification, DebertaForSequenceClassification, DebertaTokenizer, BertTokenizer, XLMForSequenceClassification, XLMTokenizer,
    XLMRobertaForSequenceClassification, XLMRobertaTokenizer, AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, precision_score, recall_score


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    precision = precision_score(
        y_true=labels, y_pred=preds, average='weighted')
    recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    return{
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "precision": precision,
        "recall": recall
    }


import csv
import string
import wordninja
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import re

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)

def clean_emoji(sen):
    sen = ''.join(c for c in sen if c <= '\uFFFF')
    return sen.replace("  ", " ")

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "I'm": "I am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "2nd": "second",
    "NY": "newyork",
    "EU": "Europe",
    "yrs": "years",
    "yoouu": "you",
    "21st": "twenty first",
    "31st": "thirty first",
    "\b1st\b": "first",
    "1st\b": " first",
    "\b4th\b": "fourth",
    "\b5th\b": "fifth",
    "\b6th\b": "sixth",
    "\b7th\b": "seventh",
    "\b8th\b": "eighth",
    "\b9th\b": "ninth",
    "\b13th\b": "thirteenth",
    "\b14th\b": "fourteenth",
    "\b15th\b": "fifteenth",
    "\b16th\b": "sixteenth",
    "\b20th\b": "twentyth",
    "YOOUU": "you",
}

# further cleaning
def clean(sen, remove_stopwords=True, contraction=True, pun=True, lemma_=False):
    sen = re.sub(r'http\S+', 'url', sen, flags=re.MULTILINE)
    sen = re.sub(r'@\S+', '@username', sen)
    sen = re.sub(r'\<a href', ' ', sen)
    sen = re.sub(r'&amp;', '', sen)
    sen = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', sen)
    sen = re.sub(r'<br />', ' ', sen)
    sen = re.sub(r"[:()]", "", sen)  # remove ()
    sen = re.sub('\s+$|^\s+', '', sen)  # remove whitespace from start of the line and end of the line
    # sen = re.sub(r'[^\x00-\x7f]', r'',
                #  sen)  # a single character in the range between  (index 0) and  (index 127) (case sensitive)
    sen = sen.strip(""" '!:?-_().,'"[]{};*""")
    sen = ' '.join([w.strip(""" '!:?-_().,'"[]{};*""") for w in re.split(' ', sen)])

    # sen = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", " NUMBER ", sen)

    # spliting words
    string = []
    for x in sen.split():
        if len(x) > 6:
            for i in wordninja.split(x):
                if len(i) > 2:
                    string.append(i)
        else:
            string.append(x)
    sen = " ".join(string)

    contraction
    new_text = []
    for word in sen.split():
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    sen = " ".join(new_text)

    # sen = re.sub(r"[^A-Za-z0-9:(),\'\`]", " ", sen)
    # sen = re.sub(r"\b\d+\b", "", sen)  #remove numbers
    sen = re.sub('\s+', ' ', sen)  # matches any whitespace characte
    # sen = re.sub(r'(?:^| )\w(?:$| )', ' ', sen).strip()  # removing single character

    # Optionally, remove stop words
    if remove_stopwords:
        sen = " ".join([i for i in sen.split() if i not in stop])

    # Optionally emove puncuations
    if pun:
        sen = ''.join(ch for ch in sen if ch not in exclude)

    # Optionally lemmatiztion
    if lemma_:
        normalized = " ".join(WordNetLemmatizer().lemmatize(word) for word in sen.split())

    return sen.strip().lower()


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    examples = []
    with open(file_path, 'r') as infile:
        lines = infile.read().strip().split('\n')
    for line in lines:
        x = line.split('\t')
        text = x[0]
        text = clean_emoji(str(text))
        # text = clean(text, remove_stopwords=True)
        label = x[1]
        examples.append({'text': text, 'label': label})
    if mode == 'test':
        for i in range(len(examples)):
            if examples[i]['text'] == 'not found':
                examples[i]['present'] = False
            else:
                examples[i]['present'] = True
    return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 tokenizer,
                                 max_seq_length=128):

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(examples):

        sentence = example['text']
        label = example['label']

        sentence_tokens = tokenizer.tokenize(sentence)[:max_seq_length - 2]
        sentence_tokens = [tokenizer.cls_token] + \
            sentence_tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)

        label = label_map[label]
        features.append({'input_ids': input_ids,
                         'label': label})
        if 'present' in example:
            features[-1]['present'] = example['present']

    return features


def get_labels(data_dir):
    all_path = os.path.join(data_dir, "all.txt")
    labels = []
    with open(all_path, "r") as infile:
        lines = infile.read().strip().split('\n')

    for line in lines:
        splits = line.split('\t')
        label = splits[-1]
        if label not in labels:
            labels.append(label)
    return labels


def train(args, train_dataset, valid_dataset, model, tokenizer, labels):

    # Prepare train data
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate)
    train_batch_size = args.train_batch_size

    # Prepare optimizer
    t_total = len(train_dataloader) * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total // 10, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", train_batch_size)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    best_f1_score = 0
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
            outputs = model(**inputs, return_dict=False)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

        # Checking for validation accuracy and stopping after drop in accuracy for 3 epochs
        results = evaluate(args, model, tokenizer, labels, 'validation')[0]
        if results.get('f1') > best_f1_score and args.save_steps > 0:
            best_f1_score = results.get('f1')
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(
                args.output_dir, "training_args.bin"))

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, mode, prefix=""):

    eval_dataset = load_and_cache_examples(args, tokenizer, labels, mode=mode)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate)
    results = {}

    # Evaluation
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[2]}
            '''print(inputs["input_ids"])
            print(inputs["attention_mask"])
            print(inputs["token_type_ids"])'''
            outputs = model(**inputs, return_dict=False)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    logits_preds = preds
    preds = np.argmax(preds, axis=1)
    if mode == "test":
        preds_list = []
        label_map = {i: label for i, label in enumerate(labels)}

        for i in range(out_label_ids.shape[0]):
            if eval_dataset[i][2] == 0:
                preds_list.append('not found')
            else:
                preds_list.append(label_map[preds[i]])

        return [preds_list, logits_preds]

    else:
        result = acc_and_f1(preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results %s *****", prefix)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        return [results, logits_preds]


class CustomDataset(Dataset):
    def __init__(self, input_ids, labels, present=None):
        self.input_ids = input_ids
        self.labels = labels
        self.present = present

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.present:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long), self.present[i]
        else:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)


def collate(examples):
    padding_value = 0

    first_sentence = [t[0] for t in examples]
    first_sentence_padded = torch.nn.utils.rnn.pad_sequence(
        first_sentence, batch_first=True, padding_value=padding_value)

    max_length = first_sentence_padded.shape[1]
    first_sentence_attn_masks = torch.stack([torch.cat([torch.ones(len(t[0]), dtype=torch.long), torch.zeros(
        max_length - len(t[0]), dtype=torch.long)]) for t in examples])

    labels = torch.stack([t[1] for t in examples])

    return first_sentence_padded, first_sentence_attn_masks, labels


def load_and_cache_examples(args, tokenizer, labels, mode):

    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file(args.data_dir, mode)
    features = convert_examples_to_features(examples, labels, tokenizer, args.max_seq_length)

    # Convert to Tensors and build dataset
    all_input_ids = [f['input_ids'] for f in features]
    all_labels = [f['label'] for f in features]
    args = [all_input_ids, all_labels]
    if 'present' in features[0]:
        present = [1 if f['present'] else 0 for f in features]
        args.append(present)

    dataset = CustomDataset(*args)
    return dataset


def main():

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # Optional Parameters
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--model_type", type=str,
                        default='bert', help='type of model xlm/xlm-roberta/bert')
    parser.add_argument("--model_name", default='bert-base-multilingual-cased',
                        type=str, help='name of pretrained model/path to checkpoint')
    parser.add_argument("--save_steps", type=int, default=1, help='set to -1 to not save model')
    parser.add_argument("--max_seq_length", default=128, type=int, help="max seq length after tokenization")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    # Set seed
    set_seed(args)

    # Prepare data
    labels = get_labels(args.data_dir)
    num_labels = len(labels)

    # Initialize model
    tokenizer_class = {"xlm": XLMTokenizer, "bert": BertTokenizer, "xlm-roberta": XLMRobertaTokenizer, "deberta": DebertaTokenizer}
    if args.model_type not in tokenizer_class.keys():
        print("Model type has to be xlm/xlm-roberta/bert")
        exit(0)
    print(args.model_type, args.model_name)
    tokenizer = tokenizer_class[args.model_type].from_pretrained(
        args.model_name, do_lower_case=True)
    model_class = {"xlm": XLMForSequenceClassification, "bert": BertForSequenceClassification, "xlm-roberta": XLMRobertaForSequenceClassification, "deberta": DebertaForSequenceClassification}
    model = model_class[args.model_type].from_pretrained(
        args.model_name, num_labels=num_labels)

    model.to(args.device)

    # Training

    logger.info("Training/evaluation parameters %s", args)

    train_dataset = load_and_cache_examples(
        args, tokenizer, labels, mode="train")
    valid_dataset = load_and_cache_examples(
        args, tokenizer, labels, mode="validation")
    global_step, tr_loss = train(
        args, train_dataset, valid_dataset, model, tokenizer, labels)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation

    results = {}

    result = evaluate(args, model, tokenizer, labels, mode="validation")
    val_logits = result[1]
    val_preds = result[0]
    temp = evaluate(args, model, tokenizer, labels, mode="test")
    logits = temp[1]
    preds = temp[0]

    # Saving predictions
    output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
    with open(output_test_predictions_file, "w") as writer:
        writer.write('\n'.join(preds))

    output_test_predictions_file = os.path.join(args.output_dir, args.model_type + "logits.txt")
    with open(output_test_predictions_file, "w") as writer:
        writer.write('\t'.join(labels))
        writer.write('\n')
        writer.write('\n'.join('\t'.join(map(str, row)) for row in logits))

    output_test_predictions_file = os.path.join(args.output_dir, args.model_type + "_val_logits.txt")
    with open(output_test_predictions_file, "w") as writer:
        writer.write('\t'.join(labels))
        writer.write('\n')
        writer.write('\n'.join('\t'.join(map(str, row)) for row in val_logits))

    return results


if __name__ == "__main__":
    main()
