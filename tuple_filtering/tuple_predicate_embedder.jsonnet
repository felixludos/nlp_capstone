{
  "dataset_reader":{
    "type": "tuple_reader",
    "lazy": true,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    }
  },
  "train_data_path": "data/aristo.txt",
  "model": {
    "type": "tuple_predicate_embedder",
    "text_field_embedder": {
      "entity_embedder": {
        "tokens": {
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
            "type": "embedding",
            "embedding_dim": 300,
            "trainable": false
        }
      },
      "predicate_embedder": {
        "tokens": {
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
            "type": "embedding",
            "embedding_dim": 300,
            "trainable": false
        }
      }
    },
    "output_layer": {
        "input_dim": 600,
        "num_layers": 2,
        "output_dims": [300, 300],
        "dropout": [0.2, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 100
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 5,
    "grad_norm": 5.0,
    # comment this out
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}