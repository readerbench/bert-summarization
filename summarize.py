import argparse
import contextlib
import copy
import os

import numpy as np
import spacy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, TFBertForMaskedLM

tf.get_logger().setLevel('ERROR')

class BertSummarizer():

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('readerbench/RoBERT-large')
        self.model = TFBertForMaskedLM.from_pretrained('readerbench/RoBERT-large')
        self.nlp = spacy.load('ro_core_news_lg')

    def get_buckets(self, split_document):
        actual_sum = 0
        buckets = []
        buckets.append([])
        no_bucket = 0
        for sentence in split_document:
            _, count_tokens = self.tokenizer.encode(sentence, return_tensors="tf").shape
            if actual_sum + count_tokens < 512:
                actual_sum += count_tokens
                buckets[no_bucket].append(sentence)
            else:
                actual_sum = 0
                no_bucket += 1
                buckets.append([])
                buckets[no_bucket].append(sentence)
        return buckets

    def get_bucket_encoding(self, sentence, bucket_with_sentence):
        input_ids = self.tokenizer([sentence, bucket_with_sentence], return_tensors="tf", padding=True)['input_ids']
        encoded_buckets = input_ids.numpy().tolist()
        return encoded_buckets

    @staticmethod
    def find_sub_list(sl,l):
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                return ind,ind+sll-1

    def masked_token_position(self, encoded_sentence, encoded_bucket_with_sentence):
        token_list = self.tokenizer.convert_ids_to_tokens(tf.convert_to_tensor(encoded_bucket_with_sentence))
        text = ' '.join(token_list)
        doc = self.nlp(text)
        nouns_list = list(set([token.text for token in doc if token.pos_ in ['PROPN', 'NOUN']]))

        sent_index_start, sent_index_end = self.find_sub_list(encoded_sentence, encoded_bucket_with_sentence)
        all_noun_indexes = np.in1d(np.array(token_list), np.array(nouns_list)).nonzero()[0]
        all_noun_indexes = all_noun_indexes[(all_noun_indexes < sent_index_start) | (all_noun_indexes > sent_index_end)]
        chosen_indexes_with_sentence = np.random.choice(all_noun_indexes, int(0.2 * len(all_noun_indexes))).tolist()

        masked_bucket_with_sentence = copy.deepcopy(encoded_bucket_with_sentence)
        for index in chosen_indexes_with_sentence:
            masked_bucket_with_sentence[index] = self.tokenizer.mask_token_id
        
        masked_bucket_without_sentence = copy.deepcopy(masked_bucket_with_sentence)
        arr = np.array(masked_bucket_without_sentence)
        arr[sent_index_start : sent_index_end + 1] = self.tokenizer.pad_token_id
        masked_bucket_without_sentence = arr.tolist()

        return (masked_bucket_with_sentence, masked_bucket_without_sentence)

    def get_prediction_score(self, masked_bucket_with_sentence, masked_bucket_without_sentence, original_encoding):
        labels = tf.where(tf.convert_to_tensor(masked_bucket_with_sentence) == self.tokenizer.mask_token_id, tf.convert_to_tensor(original_encoding), -100)
        masked_outputs_with_sentence = self.model(tf.convert_to_tensor([masked_bucket_with_sentence]), labels=labels)
        masked_prediction_scores_with_sentence = masked_outputs_with_sentence.loss

        masked_outputs_without_sentence = self.model(tf.convert_to_tensor([masked_bucket_without_sentence]), labels=labels)
        masked_prediction_scores_without_sentence = masked_outputs_without_sentence.loss


        res_with_sentence = tf.math.reduce_mean(masked_prediction_scores_with_sentence)
        res_without_sentence = tf.math.reduce_mean(masked_prediction_scores_without_sentence)
        return res_without_sentence - res_with_sentence

    def summarize(self, text):
        token_string = sent_tokenize(text)
        sentence_scores = []
        my_buckets = self.get_buckets(token_string)
        sent_index = -1
        for my_bucket in my_buckets:
            for removed_sentence_index in range(len(my_bucket)):
                sent_index += 1
                new_sentences = [my_bucket[i] for i in range(len(my_bucket)) if i != removed_sentence_index]
                
                encoding = self.get_bucket_encoding(my_bucket[removed_sentence_index], ' '.join(my_bucket))
                encoded_sentence = copy.deepcopy(encoding[0])
                encoded_sentence = [
                    enc_token 
                    for enc_token in encoded_sentence 
                    if enc_token not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]
                ]
                encoded_bucket_with_sentence = copy.deepcopy(encoding[1])


                (masked_bucket_with_sentence, masked_bucket_without_sentence) = self.masked_token_position(encoded_sentence, encoded_bucket_with_sentence)
                sentence_score = self.get_prediction_score(masked_bucket_with_sentence, masked_bucket_without_sentence, encoding[1])
                sentence_scores.append((sent_index, sentence_score, my_bucket[removed_sentence_index]))
        sentence_scores.sort(key = lambda x: x[1], reverse = True)
        sentence_scores = sentence_scores[:int(0.4 * len(sentence_scores))]
        sentence_scores.sort(key = lambda x: x[0])
        predicted_summary = [sent_score[2] for sent_score in sentence_scores]
        return " ".join(predicted_summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extractive summarization with Bert')
    parser.add_argument("input_file", type=str, help="Text file to summarize")
    parser.add_argument("output_file", type=str, help="Output file")
    args = parser.parse_args()
    
    with open(args.input_file, "rt") as f:
        text = f.read()
    with contextlib.redirect_stdout(os.devnull):
        with contextlib.redirect_stderr(os.devnull):
            summarizer = BertSummarizer()
            summary = summarizer.summarize(text)
    with open(args.output_file, "wt") as f:
        f.write(summary)