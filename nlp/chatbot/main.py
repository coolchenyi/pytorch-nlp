# -*- coding: utf-8 -*-

import argparse
from nlp.chatbot.train import train_iters
from nlp.chatbot.evaluate import run_test


def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-l', '--load', help='Load the model and train')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
    parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
    parser.add_argument('-it', '--iteration', type=int, default=1000, help='Train the model with it iterations')
    parser.add_argument('-p', '--print', type=int, default=50, help='Print every p iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
    parser.add_argument('-hi', '--hidden', type=int, default=512, help='Hidden size in encoder and decoder')
    parser.add_argument('-be', '--beam', type=int, default=1, help='Beam size of results')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-s', '--save', type=float, default=100, help='Save every s iterations')
    parser.add_argument('-sd', '--save_dir', type=str, help='Save every s iterations')

    args = parser.parse_args()
    return args


def parse_file_name(filename):
    filename = filename.split('/')
    data_type = filename[-1][:-4]   # remove '.tar'
    parse = data_type.split('_')
    reverse = 'reverse' in parse
    layers, hidden = filename[-2].split('_')
    n_layers = int(layers.split('-')[0])
    hidden_size = int(hidden)
    return n_layers, hidden_size, reverse


def run(args):
    if args.train and not args.load:
        train_iters(args.train, args.reverse, args.iteration, args.learning_rate, args.batch_size,
                   args.layer, args.hidden, args.print, args.save, args.save_dir)
    elif args.load:
        n_layers, hidden_size, reverse = parse_file_name(args.load)
        train_iters(args.train, reverse, args.iteration, args.learning_rate, args.batch_size,
                    n_layers, hidden_size, args.print, args.save, args.save_dir, load_file_name=args.load)
    elif args.test:
        n_layers, hidden_size, reverse = parse_file_name(args.test)
        run_test(n_layers, hidden_size, reverse, args.test, args.beam, True, args.corpus, args.save_dir)


if __name__ == '__main__':
    args = parse()
    run(args)