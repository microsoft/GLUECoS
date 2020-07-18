# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from tqdm import tqdm, trange
from transformers import (
    BertForTokenClassification, BertTokenizer, XLMForTokenClassification, XLMTokenizer,
    XLMRobertaForTokenClassification, XLMRobertaTokenizer, AdamW, get_linear_schedule_with_warmup
)
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    examples = []
    with open(file_path, 'r') as infile:
        lines = infile.read().strip().split('\n\n')
    for example in lines:
        example = example.split('\n')
        words = [line.split('\t')[0] for line in example]
        labels = [line.split('\t')[-1] for line in example]
        examples.append({'words': words, 'labels': labels})
    if mode == 'test':
        for i in range(len(examples)):
            if examples[i]['words'][0] == 'not found':
                examples[i]['present'] = False
            else:
                examples[i]['present'] = True
    return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 tokenizer,
                                 max_seq_length=128):

    label_map = {label: i for i, label in enumerate(label_list)}
    pad_token_label_id = CrossEntropyLoss().ignore_index
    features = []
    for (ex_index, example) in enumerate(examples):

        sentence = []
        labels = []
        for word, label in zip(example['words'], example['labels']):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                sentence.extend(word_tokens)

                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                labels.extend([label_map[label]]
                              + [pad_token_label_id] * (len(word_tokens) - 1))
                assert len(sentence) == len(labels)

        sentence_tokens = sentence[:max_seq_length - 2]
        label_ids = labels[:max_seq_length - 2]

        sentence_tokens = [tokenizer.cls_token] + \
            sentence_tokens + [tokenizer.sep_token]
        label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]

        input_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)

        assert len(input_ids) == len(label_ids)

        features.append({'input_ids': input_ids,
                         'label_ids': label_ids})
        if 'present' in example:
            features[-1]['present'] = example['present']
    return features


def get_labels(data_dir):
    all_path = os.path.join(data_dir, "all.txt")
    labels = []
    with open(all_path, "r") as infile:
        lines = infile.read().strip().split('\n\n')

    for example in lines:
        example = example.split('\n')
        for label in [e.split('\t')[-1] for e in example]:
            if label not in labels:
                labels.append(label)
    return labels


def train(args, train_dataset, valid_dataset, model, tokenizer, labels):

    # Prepare train data
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate)

    # Prepare optimizer
    t_total = len(train_dataloader) * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total // 10, num_training_steps=t_total)

    # Training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size)

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
            inputs = {"input_ids": batch[0],
                      'attention_mask': batch[1],
                      "labels": batch[2]}
            outputs = model(**inputs)
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
        results = evaluate(args, model, tokenizer, labels, 'validation')
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

    # Evaluation
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    out_label_ids = []
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[2]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1

        preds.extend([t for t in logits.detach().cpu()])
        out_label_ids.extend([t for t in inputs["labels"].detach().cpu()])

    eval_loss = eval_loss / nb_eval_steps
    preds = [np.argmax(t, axis=1) for t in preds]

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(len(out_label_ids))]
    preds_list = [[] for _ in range(len(out_label_ids))]

    pad_token_label_id = CrossEntropyLoss().ignore_index
    for i in range(len(out_label_ids)):
        for j in range(out_label_ids[i].shape[0]):
            if out_label_ids[i][j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j].item()])
                preds_list[i].append(label_map[preds[i][j].item()])

    if mode == "test":
        for i in range(len(preds_list)):
            if eval_dataset[i][2] == 0:
                preds_list[i] = ['not found']
        return preds_list
    else:
        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
            "accuracy": accuracy_score(out_label_list, preds_list)
        }

        logger.info("***** Eval results %s *****", prefix)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results


class CustomDataset(Dataset):
    def __init__(self, input_ids, label_ids, present=None):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.present = present

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, i):
        if self.present:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.label_ids[i], dtype=torch.long), self.present[i]
        else:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.label_ids[i], dtype=torch.long)


def collate(examples):
    padding_value = 0

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    first_sentence = [t[0] for t in examples]
    first_sentence_padded = torch.nn.utils.rnn.pad_sequence(
        first_sentence, batch_first=True, padding_value=padding_value)

    max_length = first_sentence_padded.shape[1]
    first_sentence_attn_masks = torch.stack([torch.cat([torch.ones(len(t[0]), dtype=torch.long), torch.zeros(
        max_length - len(t[0]), dtype=torch.long)]) for t in examples])

    labels = torch.stack([torch.cat([t[1], torch.tensor(
        [pad_token_label_id] * (max_length - len(t[1])), dtype=torch.long)]) for t in examples])

    return first_sentence_padded, first_sentence_attn_masks, labels


def load_and_cache_examples(args, tokenizer, labels, mode):

    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file(args.data_dir, mode)
    features = convert_examples_to_features(examples, labels, tokenizer, args.max_seq_length)

    # Convert to Tensors and build dataset
    all_input_ids = [f['input_ids'] for f in features]
    all_label_ids = [f['label_ids'] for f in features]
    args = [all_input_ids, all_label_ids]
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
                        default='bert', help='type of model xlm-roberta/bert')
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

    tokenizer_class = {"xlm": XLMTokenizer, "bert": BertTokenizer, "xlm-roberta": XLMRobertaTokenizer}
    if args.model_type not in tokenizer_class.keys():
        print("Model type has to be xlm/xlm-roberta/bert")
        exit(0)
    tokenizer = tokenizer_class[args.model_type].from_pretrained(
        args.model_name, do_lower_case=True)
    model_class = {"xlm": XLMForTokenClassification, "bert": BertForTokenClassification, "xlm-roberta": XLMRobertaForTokenClassification}
    model = model_class[args.model_type].from_pretrained(
        args.model_name, num_labels=num_labels)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    train_dataset = load_and_cache_examples(
        args, tokenizer, labels, mode="train")
    valid_dataset = load_and_cache_examples(
        args, tokenizer, labels, mode="validation")
    global_step, tr_loss = train(
        args, train_dataset, valid_dataset, model, tokenizer, labels)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`

    # Evaluation
    results = {}
    result = evaluate(args, model, tokenizer, labels, mode="validation")
    preds = evaluate(args, model, tokenizer, labels, mode="test")

    # Saving predictions
    output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
    with open(output_test_predictions_file, "w") as writer:
        writer.write('\n\n'.join(['\n'.join(example) for example in preds]))

    return results


if __name__ == "__main__":
    main()
