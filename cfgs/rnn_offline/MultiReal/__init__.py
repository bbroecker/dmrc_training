class AngleConfig(object):
    init_scale = 0.01
    # init_scale = 1.0
    max_learning_rate = 0.001
    min_learning_rate = 0.001
    learning_rate_time = 120.0
    max_grad_norm = 30
    num_layers = 2
    state_count = 5
    # state_count = 3
    trace_length = 30
    hidden_size_rnn = [200]
    hidden_size_dense = [50, 10]
    # max_epoch = 6
    # max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 40
    use_tanh_rnn = [False]
    use_tanh_dense = [False, False]
    buffer_size = 500000
    forget_bias = 0.0
    use_fp16 = False
    output_size = 16
    valid_batch_size = 300
    log_folder = None
    weight_folder = None
    # weight_folder = "./weights_save/rnn_cnn/angle_real/batch/lr0.001_tanh900__batch30_trace_length6_forget_bias0.0_init_value0.01_keep_prob1.0_AdamOptimizer_1/"
    # vocab_size = 10000
    # optimizer = "GradientDescentOptimizer"
    optimizer = "AdamOptimizer"
    # optimizer = "RMSPropOptimize