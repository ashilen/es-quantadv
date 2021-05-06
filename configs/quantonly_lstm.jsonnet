{
  dataset_reader: {
    type: 'uds_time_reader',
    token_indexers: {
      tokens: {
        type: 'single_id',
        namespace: 'tokens',
        lowercase_tokens: true
      }
    }
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 10
    }
  },
  train_data_path: 'data/quantonly/train',
  validation_data_path: 'data/quantonly/dev',
  test_data_path: 'data/quantonly/test',
  model: {
    type: 'containment_to_quant_classifier',
    embedder: {
      token_embedders: {
        tokens: {
        type: 'embedding',
          pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
          embedding_dim: 50,
          trainable: false
        }
      }
    },
    encoder: {
      type: 'lstm',
      input_size: 200,
      hidden_size: 25,
      bidirectional: true
    }
  },
  trainer: {
    num_epochs: 40,
    patience: 10,
    cuda_device: -1,
    grad_clipping: 5.0,
    validation_metric: '-loss',
    optimizer: {
      type: 'adam',
      lr: 0.003
    }
  }
}
