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

## 3 Translate

```bash
python translate.py -model demo-model_acc_ppl_epoch.pt -src data/nmt/src-test.txt -output pred.txt -replace_unk -verbose
```