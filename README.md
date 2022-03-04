# Novel Searcher ("Project Under Development")

---



Nobel searcher is a document search application using colBERT.

## Preprocess data

```
python utils/preprocess_raw.py 
```

## Training
Training requires a list of <query, positive passage, negative passage> tab-separated triples.

```
python train.py --train_file=/path/to/train_file 
```
## Retrieval

### Construct index
Index must be created in advance before retrieval. Index uses faiss to store indexes for each document.
```
python index.py --checkpoint_path=/path/to/*checkpoint_directory* \
                --document_paths=/path/to/*document_directory* /path/to/*document_file* ...  \
                --index_dir=/path/to/*index_dir_to_save* \
```

### Do retrieval

```
python retrieval.py --checkpoint_path=/path/to/*checkpoint_directory* \
                    --document_name="<document_name>" \
                    --index_dir=/path/to/*index_dir_to_save* \
                    --query="<query>" \
```

## Evaluate

```
python evaluate.py --checkpoint_path=/path/to/*checkpoint_directory* \
                   --document_name="<document_name>" \
                   --index_dir=/path/to/*index_dir_to_save* \
                   --query="<query>" \
```

## Reference

- [ColBERT](https://github.com/stanford-futuredata/ColBERT)