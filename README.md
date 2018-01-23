# pytorch-nlp

## 1 word embedding

```bash
python run.py
```

## 2 named entity recognition

### 2.1 Train the model

```bash
python train.py
```

## 3 machine translation

### 3.1 Preprocess the data

```bash
python preprocess.py -train_src data/nmt/src-train.txt -train_tgt data/nmt/tgt-train.txt -valid_src data/nmt/src-val.txt
 -valid_tgt data/nmt/tgt-val.txt -save_data data/nmt/demo
```

### 3.2 Train the model

```bash
python train.py -data data/nmt/demo -save_model demo-model
```

### 3.3 Translate

```bash
python translate.py -model demo-model_epochX_PPL.pt -src data/nmt/src-test.txt -output pred.txt -replace_unk -verbose
```

## 4 chatbot

update soon ...

## 5 speech recognition

update soon ...

## 6 text to speech

update soon ...