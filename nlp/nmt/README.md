# Neural Machine Translation

## 1 Preprocess the data

```bash
python preprocess.py -train_src data/nmt/src-train.txt -train_tgt data/nmt/tgt-train.txt -valid_src data/nmt/src-val.txt
 -valid_tgt data/nmt/tgt-val.txt -save_data data/nmt/demo
```

## 2 Train the model

```bash
python train.py -data data/nmt/demo -save_model demo-model
```
or you can use the following code to train model:

```python
import torch
import torch.nn as nn

from nlp.nmt.onmt.io.DatasetBase import PAD_WORD
from nlp.nmt.onmt import Trainer
from nlp.nmt.onmt import Loss
from nlp.nmt.onmt import Optim
from nlp.nmt.onmt.io.IO import OrderedIterator
from nlp.nmt.onmt.modules.Embeddings import Embeddings
from nlp.nmt.onmt.Models import RNNEncoder, InputFeedRNNDecoder, NMTModel



vocab = dict(torch.load("../../data/data.vocab.pt"))
src_padding = vocab["src"].stoi[PAD_WORD]
tgt_padding = vocab["tgt"].stoi[PAD_WORD]

emb_size = 10
rnn_size = 6
# Specify the core model.
encoder_embeddings = Embeddings(emb_size, len(vocab["src"]),
                                             word_padding_idx=src_padding)

encoder = RNNEncoder(hidden_size=rnn_size, num_layers=1,
                                 rnn_type="LSTM", bidirectional=True,
                                 embeddings=encoder_embeddings)

decoder_embeddings = Embeddings(emb_size, len(vocab["tgt"]),
                                             word_padding_idx=tgt_padding)
decoder = InputFeedRNNDecoder(hidden_size=rnn_size, num_layers=1,
                                           bidirectional_encoder=True,
                                           rnn_type="LSTM", embeddings=decoder_embeddings)
model = NMTModel(encoder, decoder)

# Specify the tgt word generator and loss computation module
model.generator = nn.Sequential(
            nn.Linear(rnn_size, len(vocab["tgt"])),
            nn.LogSoftmax())
loss = Loss.NMTLossCompute(model.generator, vocab["tgt"])

optim = Optim.Optim(method="sgd", lr=1, max_grad_norm=2)
optim.set_parameters(model.parameters())

# Load some data
data = torch.load("../../data/data.train.pt")
valid_data = torch.load("../../data/data.valid.pt")
data.load_fields(vocab)
valid_data.load_fields(vocab)
data.examples = data.examples[:100]

train_iter = OrderedIterator(
                dataset=data, batch_size=10,
                device=-1,
                repeat=False)
valid_iter = OrderedIterator(
                dataset=valid_data, batch_size=10,
                device=-1,
                train=False)
                
trainer = Trainer.Trainer(model, train_iter, valid_iter, loss, loss, optim=optim)

def report_func(*args):
    stats = args[-1]
    stats.output(args[0], args[1], 10, 0)
    return stats

for epoch in range(2):
    trainer.train(epoch, report_func)
    val_stats = trainer.validate(valid_iter)

    print("Validation")
    val_stats.output(epoch, 11, 10, 0)
    trainer.epoch_step(val_stats.ppl(), epoch)
```

## 3 Translate

```bash
python translate.py -model demo-model_acc_ppl_epoch.pt -src data/nmt/src-test.txt -output pred.txt -replace_unk -verbose
```
or

```python
from nlp.nmt.onmt.translate import Translator, Translation

translator = Translator.Translator(beam_size=10, fields=data.fields, model=model)
builder = Translation.TranslationBuilder(data=valid_data, fields=data.fields)

for batch in valid_iter:
    trans_batch = translator.translate_batch(batch=batch, data=valid_data)
    translations = builder.from_batch(trans_batch)
    for trans in translations:
        print(trans.log(0))
    break
```