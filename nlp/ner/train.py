# -*- coding: utf-8 -*-

import optparse
from collections import OrderedDict
import torch
import time
from torch.autograd import Variable
from nlp.ner.eval import return_report
from nlp.ner.utils import *
from nlp.ner.loader import *
from nlp.ner.model import BiLSTM_CRF
from nlp.ner.preprocess import preprocess_data

t = time.time()
models_path = "models/"

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)


def train():
    word_to_id, tag_to_id, id_to_tag, word_embeds, train_data, dev_data, test_data =\
        preprocess_data(parameters, opts, mapping_file=mapping_file)

    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    model = BiLSTM_CRF(vocab_size=len(word_to_id),
                       tag_to_ix=tag_to_id,
                       embedding_dim=parameters['word_dim'],
                       hidden_dim=parameters['word_lstm_dim'],
                       use_gpu=use_gpu,
                       # char_to_ix=char_to_id,
                       pre_word_embeds=word_embeds,
                       use_crf=parameters['crf'])
    if parameters['reload']:
        model.load_state_dict(torch.load(model_name))
    if use_gpu:
        model.cuda()
    learning_rate = 0.015
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss = 0.0
    best_dev_F = -1.0
    best_test_F = -1.0
    plot_every = 10
    eval_every = 20
    count = 0

    model.train(True)
    for epoch in range(1, 10001):
        print("train epoch %i :" % epoch)
        for i, index in enumerate(np.random.permutation(len(train_data))):
            tr = time.time()
            count += 1
            data = train_data[index]
            model.zero_grad()

            sentence_in = data['words']
            sentence_in = Variable(torch.LongTensor(sentence_in))
            tags = data['tags']

            # char lstm
            # if parameters['char_mode'] == 'LSTM':
            #     chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            #     d = {}
            #     for i, ci in enumerate(chars2):
            #         for j, cj in enumerate(chars2_sorted):
            #             if ci == cj:
            #                 d[j] = i
            #                 continue
            #     chars2_length = [len(c) for c in chars2_sorted]
            #     char_maxl = max(chars2_length)
            #     chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            #     for i, c in enumerate(chars2_sorted):
            #         chars2_mask[i, :chars2_length[i]] = c
            #     chars2_mask = Variable(torch.LongTensor(chars2_mask))

            # char cnn
            # if parameters['char_mode'] == 'CNN':
            #     d = {}
            #     chars2_length = [len(c) for c in chars2]
            #     char_max_l = max(chars2_length)
            #     chars2_mask = np.zeros((len(chars2_length), char_max_l), dtype='int')
            #     for i, c in enumerate(chars2):
            #         chars2_mask[i, :chars2_length[i]] = c
            #     chars2_mask = Variable(torch.LongTensor(chars2_mask))

            targets = torch.LongTensor(tags)
            # caps = Variable(torch.LongTensor(data['caps']))
            if use_gpu:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda())
            else:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)
            loss += neg_log_likelihood.data[0] / len(data['words'])
            neg_log_likelihood.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()

            if count % plot_every == 0:
                loss /= plot_every
                print(count, ': ', loss)
                # if not losses:
                #     losses.append(loss)
                # losses.append(loss)
                # text = '<p>' + '</p><p>'.join([str(l) for l in losses[-9:]]) + '</p>'
                # loss_win = 'loss_' + name
                # text_win = 'loss_text_' + name
                # vis.line(np.array(losses), X=np.array([plot_every * i for i in range(len(losses))]),
                #          win=loss_win, opts={'title': loss_win, 'legend': ['loss']})
                # vis.text(text, win=text_win, opts={'title': text_win})
                loss = 0.0

            if count % eval_every == 0 and count > (eval_every * 20) or \
               count % (eval_every * 4) == 0 and count < (eval_every * 20):
                model.train(False)
                # best_train_F, new_train_F, _ = evaluating(model, test_train_data, best_train_F, tag_to_id, id_to_tag)
                best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F, tag_to_id, id_to_tag)
                if save:
                    torch.save(model, model_name)
                # best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F, tag_to_id, id_to_tag)
                # sys.stdout.flush()

                # all_F.append([new_train_F, new_dev_F, new_test_F])
                # Fwin = 'F-score of {train, dev, test}_' + name
                # vis.line(np.array(all_F), win=Fwin,
                #          X=np.array([eval_every * i for i in range(len(all_F))]),
                #          opts={'title': Fwin, 'legend': ['train', 'dev', 'test']})
                model.train(True)

            if count % len(train_data) == 0:
                adjust_learning_rate(optimizer, lr=learning_rate / (1 + 0.05 * count / len(train_data)))

    print(time.time() - t)


def evaluating(model, datas, best_F, tag_to_id, id_to_tag):
    prediction = []
    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        # chars2 = data['chars']
        # caps = data['caps']

        # if parameters['char_mode'] == 'LSTM':
        #     chars_2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
        #     d = {}
        #     for i, ci in enumerate(chars2):
        #         for j, cj in enumerate(chars_2_sorted):
        #             if ci == cj:
        #                 d[j] = i
        #                 continue
        #     chars2_length = [len(c) for c in chars_2_sorted]
        #     char_maxl = max(chars2_length)
        #     chars2_mask = np.zeros((len(chars_2_sorted), char_maxl), dtype='int')
        #     for i, c in enumerate(chars_2_sorted):
        #         chars2_mask[i, :chars2_length[i]] = c
        #     chars2_mask = Variable(torch.LongTensor(chars2_mask))
        #
        # if parameters['char_mode'] == 'CNN':
        #     d = {}
        #     chars2_length = [len(c) for c in chars2]
        #     char_maxl = max(chars2_length)
        #     chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
        #     for i, c in enumerate(chars2):
        #         chars2_mask[i, :chars2_length[i]] = c
        #     chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        # dcaps = Variable(torch.LongTensor(caps))
        if use_gpu:
            val, out = model(dwords.cuda())
        else:
            val, out = model(dwords)
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
    predf = eval_temp + '/pred.' + name
    # scoref = eval_temp + '/score.' + name

    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    eval_lines = return_report(predf)

    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_F = float(line.strip().split()[-1])
            if new_F > best_F:
                best_F = new_F
                save = True
                print('the best F is ', new_F)

    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    ))
    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        ))
    return best_F, new_F, save


if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option(
        "-T", "--train", default="/Users/endy/nlp/pytorch-nlp/data/ner/example.dev",
        help="Train set location"
    )
    optparser.add_option(
        "-d", "--dev", default="/Users/endy/nlp/pytorch-nlp/data/ner/example.dev",
        help="Dev set location"
    )
    optparser.add_option(
        "-t", "--test", default="/Users/endy/nlp/pytorch-nlp/data/ner/example.test",
        help="Test set location"
    )
    optparser.add_option(
        '--test_train', default='/Users/endy/nlp/pytorch-nlp/data/ner/example.test',
        help='test train'
    )
    optparser.add_option(
        '--score', default='evaluation/temp/score.txt',
        help='score file location'
    )
    optparser.add_option(
        "-s", "--tag_scheme", default="iobes",
        help="Tagging scheme (IOB or IOBES)"
    )
    optparser.add_option(
        "-l", "--lower", default="1",
        type='int', help="Lowercase words (this will not affect character inputs)"
    )
    optparser.add_option(
        "-z", "--zeros", default="0",
        type='int', help="Replace digits with 0"
    )

    optparser.add_option(
        "-w", "--word_dim", default="100",
        type='int', help="Token embedding dimension"
    )
    optparser.add_option(
        "-W", "--word_lstm_dim", default="200",
        type='int', help="Token LSTM hidden layer size"
    )
    optparser.add_option(
        "-B", "--word_bidirect", default="1",
        type='int', help="Use a bidirectional LSTM for words"
    )
    optparser.add_option(
        "-p", "--pre_emb", default="/Users/endy/nlp/pytorch-nlp/data/pre_trained/vec.txt",
        help="Location of pretrained embeddings"
    )
    optparser.add_option(
        "-A", "--all_emb", default="1",
        type='int', help="Load all embeddings"
    )
    optparser.add_option(
        "-a", "--cap_dim", default="0",
        type='int', help="Capitalization feature dimension (0 to disable)"
    )
    optparser.add_option(
        "-f", "--crf", default="0",
        type='int', help="Use CRF (0 to disable)"
    )
    optparser.add_option(
        "-D", "--dropout", default="0.5",
        type='float', help="Droupout on the input (0 = no dropout)"
    )
    optparser.add_option(
        "-r", "--reload", default="0",
        type='int', help="Reload the last saved model"
    )
    optparser.add_option(
        "-g", '--use_gpu', default='1',
        type='int', help='whether or not to ues gpu'
    )
    optparser.add_option(
        '--loss', default='loss.txt',
        help='loss file location'
    )
    optparser.add_option(
        '--name', default='test',
        help='model name'
    )
    # optparser.add_option(
    #     '--char_mode', choices=['CNN', 'LSTM'], default='LSTM',
    #     help='char_CNN or char_LSTM'
    # )
    opts = optparser.parse_args()[0]

    parameters = OrderedDict()
    parameters['tag_scheme'] = opts.tag_scheme
    parameters['lower'] = opts.lower == 1
    parameters['zeros'] = opts.zeros == 1
    # parameters['char_dim'] = opts.char_dim
    # parameters['char_lstm_dim'] = opts.char_lstm_dim
    # parameters['char_bidirect'] = opts.char_bidirect == 1
    parameters['word_dim'] = opts.word_dim
    parameters['word_lstm_dim'] = opts.word_lstm_dim
    parameters['word_bidirect'] = opts.word_bidirect == 1
    parameters['pre_emb'] = opts.pre_emb
    parameters['all_emb'] = opts.all_emb == 1
    parameters['cap_dim'] = opts.cap_dim
    parameters['crf'] = opts.crf == 1
    parameters['dropout'] = opts.dropout
    parameters['reload'] = opts.reload == 1
    parameters['name'] = opts.name
    # parameters['char_mode'] = opts.char_mode

    parameters['use_gpu'] = opts.use_gpu == 1 and torch.cuda.is_available()
    use_gpu = parameters['use_gpu']

    mapping_file = 'models/mapping.pkl'

    name = parameters['name']
    model_name = models_path + name  # get_name(parameters)
    tmp_model = model_name + '.tmp'

    assert os.path.isfile(opts.train)
    assert os.path.isfile(opts.dev)
    assert os.path.isfile(opts.test)
    # assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
    assert 0. <= parameters['dropout'] < 1.
    assert parameters['tag_scheme'] in ['iob', 'iobes']
    assert not parameters['all_emb'] or parameters['pre_emb']
    assert not parameters['pre_emb'] or parameters['word_dim'] > 0
    assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

    train()

