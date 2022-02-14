# Novel Searcher ("Project Under Development")

---



Nobel searcher is a document search application using colBERT.

## Preprocess data

```
python utils/preprocess_raw.py 
```

## Training


```
python train.py --train_file=/path/to/train_file 
```
## Retrieval

### Construct index
Index must be created in advance before retrieval. Index uses faiss to store indexes for each document.
```

```

### Do retrieval

```

```

## Reference

- [ColBERT](https://github.com/stanford-futuredata/ColBERT)