# pytorch-nlp

## 1 word embedding

train word2vec: 

```bash
cd pytorch-nlp/nlp/word-embeddings

 python run.py --input_file_name=path/to/data_file \
        --output_file_name=path/to/data_dir \
        --method=cbow or skip-gram \
        --emb_dimension=100 \
        --batch_size=100 \
        --window_size=2 \
        --iteration=10 \
        --initial_lr=0.025 \
        --min_count=5 \
        --using_hs=False \
        --using_neg=False \
        --num_threads=5 \
        --context_size=5
```


## 参考

* https://github.com/bamtercelboo/pytorch_word2vec
