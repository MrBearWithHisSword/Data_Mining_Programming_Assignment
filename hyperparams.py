class Hyperparams:
    # data
    train_data_dir = 'data_dir/train_bunch_mul.dat'
    sentence_len = 80
    num_class = 11
    num_examples = 1715
    training_examples = 1372
    embedding_dim = 300
    embedding_matrix = None
    X_test = None
    Y_test = None
    load_embedding_matrix = False

    # training
    batch_size = 32
    learning_rate = 0.0001
    logdir = 'logdir'

    # model
    num_epochs = 20
    # num_blocks = 6
    num_blocks = 2
    # num_heads = 8
    num_heads = 5
    # hidden_units = 512
    hidden_units = 300
    dropout_rate = 0.1


print(Hyperparams.embedding_matrix)