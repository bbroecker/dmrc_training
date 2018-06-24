import yaml


class NNConfigAngle(object):

    def __init__(self, config_file):
        self.init_scale = None
        self.logging_freq = None
        self.init_constant = None
        self.max_learning_rate = None
        self.min_learning_rate = None
        self.learning_rate_time = None
        self.max_grad_norm = None
        self.num_layers = None
        self.state_count = None
        self.trace_length = None
        self.hidden_size_dense = None
        self.hidden_size_rnn = None

        # max_epoch = 6
        # max_max_epoch = 39
        self.keep_prob = None
        self.lr_decay = None
        self.batch_size = None
        self.use_tanh_rnn = None
        self.use_tanh_dense = None
        self.buffer_size = None
        self.forget_bias = None
        self.use_fp16 = None
        self.valid_batch_size = None
        self.store_every_ts = None
        self.train_freq = None
        self.clamp = None
        self.weight_folder = None
        self.log_folder = None
        self.min_update_freq = None
        self.max_update_freq = None
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.max_velocity = None
        self.acceleration = None
        self.damping_per_sec = None
        self.max_sensor_distance = None
        self.pre_train_episodes = None
        self.optimizer = None
        self.min_history = None
        self.update_length = None
        self.accuracy_length = None
        self.best_result = None
        self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                cfg = yaml.load(stream)['neural_network']
            except yaml.YAMLError as exc:
                print "didn't load"
                print(exc)

        self.max_velocity = cfg['dynamics']['max_velocity']
        self.acceleration = cfg['dynamics']['acceleration']
        self.damping_per_sec = cfg['dynamics']['damping_per_sec']
        self.max_sensor_distance = cfg['sensors']['max_sensor_distance']
        self.update_rate = cfg['controls']['update_rate']
        self.init_scale = cfg['network']['init_scale']
        self.num_layers = cfg['network']['num_layers']
        self.state_count = cfg['network']['state_count']
        self.init_constant = cfg['network']['init_constant']
        self.logging_freq = cfg['network']['logging_freq']
        self.max_learning_rate = cfg['network']['max_learning_rate']
        self.min_learning_rate = cfg['network']['min_learning_rate']
        self.learning_rate_time = cfg['network']['learning_rate_time']
        self.max_grad_norm = cfg['network']['max_grad_norm']
        self.trace_length = cfg['network']['trace_length']
        self.hidden_size_dense = cfg['network']['hidden_size_dense']
        self.hidden_size_rnn = cfg['network']['hidden_size_rnn']
        #only interesting for descrete angles
        self.output_size = cfg['network']['output_size']
        self.min_history = cfg['network']['min_history']
        self.update_length = cfg['network']['update_length']
        self.accuracy_length = cfg['network']['accuracy_length']

        # max_epoch = 6
        # max_max_epoch = 39
        self.keep_prob = cfg['network']['keep_prob']
        self.lr_decay = cfg['network']['lr_decay']
        self.batch_size = cfg['network']['batch_size']
        self.use_tanh_dense = cfg['network']['use_tanh_dense']
        self.use_tanh_rnn = cfg['network']['use_tanh_rnn']
        self.buffer_size = cfg['network']['buffer_size']
        self.forget_bias = cfg['network']['forget_bias']
        self.use_fp16 = cfg['network']['use_fp16']
        self.valid_batch_size = cfg['network']['valid_batch_size']
        self.store_every_ts = cfg['network']['store_every_ts']
        self.train_freq = cfg['network']['train_freq']
        self.clamp = cfg['network']['clamp']
        self.weight_folder = cfg['network']['weight_folder']
        self.log_folder = cfg['network']['log_folder']
        self.min_update_freq = cfg['network']['min_update_freq']
        self.max_update_freq = cfg['network']['max_update_freq']
        self.min_x = cfg['network']['min_x']
        self.max_x = cfg['network']['max_x']
        self.min_y = cfg['network']['min_y']
        self.max_y = cfg['network']['max_y']
        self.pre_train_episodes = cfg['network']['pre_train_episodes']
        self.optimizer = cfg['network']['optimizer']
        self.best_result = cfg['network']['best_result']
        self.bucket_types = cfg['network']['bucket_types']
        self.t_bucketsize = cfg['network']['t_bucketsize']
