# -*- encoding: utf-8 -*-

from input_data import InputData
from model import SkipGramModel
from model import CBOW
import torch.optim as optim
from tqdm import tqdm


class Word2Vec:
    def __init__(self,
                 input_file_name,
                 output_file_name,
                 emb_dimension,
                 batch_size,
                 window_size,
                 iteration,
                 initial_lr,
                 min_count,
                 using_hs,
                 using_neg,
                 context_size,
                 hidden_size,
                 cbow,
                 skip_gram):
        """Initilize class parameters.
        Args:
            input_file_name: Name of a text data from file. Each line is a sentence splited with space.
            output_file_name: Name of the final embedding file.
            emb_dimention: Embedding dimention, typically from 50 to 500.
            batch_size: The count of word pairs for one forward.
            window_size: Max skip length between words.
            iteration: Control the multiple training iterations.
            initial_lr: Initial learning rate.
            min_count: The minimal word frequency, words with lower frequency will be filtered.
            using_hs: Whether using hierarchical softmax.
        Returns:
            None.
        """
        print("\nInput File loading......\n")
        self.data = InputData(input_file_name, min_count)
        print("\nInput File loaded.\n")
        print("Input Data", self.data)
        self.output_file_name = output_file_name
        self.emb_size = len(self.data.word2id)
        print("emb_size", self.emb_size)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.using_hs = using_hs
        self.using_neg = using_neg
        self.cbow = cbow
        self.skip_gram = skip_gram
        if self.skip_gram is not None and self.skip_gram:
            self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
            print("skip_gram_model", self.skip_gram_model)
            self.optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)
        if self.cbow is not None and self.cbow:
            # self.cbow_model = CBOW(self.emb_size, self.context_size, self.emb_dimension, self.hidden_size)
            self.cbow_model = CBOW(self.emb_size, self.emb_dimension)
            print("CBOW_model", self.cbow_model)
            self.optimizer = optim.SGD(self.cbow_model.parameters(), lr=self.initial_lr)

    def skip_gram_train(self):
        """Multiple training.
        Returns:
            None.
        """
        print("Skip_gram Training......")
        pair_count = self.data.evaluate_pair_count(self.window_size)
        print("pair_count", pair_count)
        batch_count = self.iteration * pair_count / self.batch_size
        print("batch_count", batch_count)
        process_bar = tqdm(range(int(batch_count)))
        for i in process_bar:
            pos_pairs = self.data.get_batch_pairs(self.batch_size, self.window_size)
            if self.using_hs:
                pos_pairs, neg_pairs = self.data.get_pairs_by_huffman(pos_pairs)
            else:
                pos_pairs, neg_pairs = self.data.get_pairs_by_neg_sampling(pos_pairs, 5)

            pos_u = [int(pair[0]) for pair in pos_pairs]
            pos_v = [int(pair[1]) for pair in pos_pairs]
            neg_u = [int(pair[0]) for pair in neg_pairs]
            neg_v = [int(pair[1]) for pair in neg_pairs]

            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_u, neg_v)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" % (loss.data[0], self.optimizer.param_groups[0]['lr']))
            print("Loss: %0.8f, lr: %0.6f" % (loss.data[0], self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        print("Skip_gram Trained and Saving File......")
        self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
        print("Skip_gram Trained and Saved File.")

    def cbow_train(self):
        print("CBOW Training......")
        pair_count = self.data.evaluate_pair_count(self.context_size * 2 + 1)
        print("pair_count", pair_count)
        batch_count = self.iteration * pair_count / self.batch_size
        print("batch_count", batch_count)
        process_bar = tqdm(range(int(batch_count)))
        for i in process_bar:
            pos_pairs = self.data.get_cbow_batch_all_pairs(self.batch_size, self.context_size)
            if self.using_hs:
                pos_pairs, neg_pairs = self.data.get_cbow_pairs_by_huffman(pos_pairs)
            else:
                pos_pairs, neg_pairs = self.data.get_cbow_pairs_by_neg_sampling(pos_pairs, self.context_size)

            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [int(pair[1]) for pair in pos_pairs]
            neg_u = [pair[0] for pair in neg_pairs]
            neg_v = [int(pair[1]) for pair in neg_pairs]

            self.optimizer.zero_grad()
            loss = self.cbow_model.forward(pos_u, pos_v, neg_u, neg_v)
            loss.backward()
            self.optimizer.step()
            process_bar.set_description("Loss: %0.8f, lr: %0.6f" % (loss.data[0], self.optimizer.param_groups[0]['lr']))
            print("Loss: %0.8f, lr: %0.6f" % (loss.data[0], self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        print("CBOW Trained and Saving File......")
        self.cbow_model.save_embedding(self.data.id2word, self.output_file_name)
        print("CBOW Trained and Saved File.")