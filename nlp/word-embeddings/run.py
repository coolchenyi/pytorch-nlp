# -*- coding:utf-8 -*-

import torch
from word2vec import Word2Vec
import argparse

def main(args):
    torch.set_num_threads(5)
    if args.method == 'cbow':
        word2vec = Word2Vec(input_file_name=args.input_file_name,
                            output_file_name=args.output_file_name,
                            emb_dimension=args.emb_dimension,
                            batch_size=args.batch_size,
                            # windows_size used by Skip-Gram model
                            window_size=args.window_size,
                            iteration=args.iteration,
                            initial_lr=args.initial_lr,
                            min_count=args.min_count,
                            using_hs=args.using_hs,
                            using_neg=args.using_neg,
                            # context_size used by CBOW model
                            context_size=args.context_size,
                            hidden_size=args.hidden_size,
                            cbow=True,
                            skip_gram=False)
        word2vec.cbow_train()
    elif args.method == 'skip_gram':
        word2vec = Word2Vec(input_file_name=args.input_file_name,
                            output_file_name=args.output_file_name,
                            emb_dimension=args.emb_dimension,
                            batch_size=args.batch_size,
                            # windows_size used by Skip-Gram model
                            window_size=args.window_size,
                            iteration=args.iteration,
                            initial_lr=args.initial_lr,
                            min_count=args.min_count,
                            using_hs=args.using_hs,
                            using_neg=args.using_neg,
                            # context_size used by CBOW model
                            context_size=args.context_size,
                            hidden_size=args.hidden_size,
                            cbow=False,
                            skip_gram=True)
        word2vec.skip_gram_train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_name', type=str,
                        default='/Users/endy/nlp/pytorch-nlp/data/word2vec/train.txt')
    parser.add_argument('--output_file_name', type=str,
                        default='/Users/endy/nlp/pytorch-nlp/data/word2vec/word2vec.txt')

    parser.add_argument('--method', type=str, default='skip_gram')

    parser.add_argument('--emb_dimension', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--iteration', type=int, default=2)
    parser.add_argument('--initial_lr', type=float, default=0.025)
    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=128)

    parser.add_argument('--using_hs', type=bool, default=False)
    parser.add_argument('--using_neg', type=bool, default=False)

    parser.add_argument('--num_threads', type=int, default=5)
    parser.add_argument('--context_size', type=int, default=5)

    args = parser.parse_args()

    main(args)
