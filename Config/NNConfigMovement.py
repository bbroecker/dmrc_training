import yaml


class NNConfigMovement(object):

    def __init__(self, config_file):
        self.init_scale = None
        self.logging_freq = None
        self.init_constant = None
        self.max_learning_rate = None
        self.min_learning_rate = None
        self.learning_rate_time = None
        self.max_grad_norm = None
        self.trace_length = None
        self.hidden_size_dense = None
        self.hidden_size_rnn = None
        # max_epoch = 6
        # max_max_epoch = 39
        self.keep_prob = None
        self.lr_decay = None
        self.batch_size = None
        self.use_tanh_dense = None
        self.buffer_size = None
        self.forget_bias = None
        self.use_fp16 = None
        self.valid_batch_size = None
        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_time = None
        self.epsilon = None
        self.gamma = None
        self.store_every_ts = None
        self.train_freq = None
        self.clamp = None
        self.weight_folder = None
        self.log_folder = None
        self.alpha = None
        self.beta = None
        self.use_QDQN = None
        self.use_double_q = None
        self.min_update_freq = None
        self.max_update_freq = None
        self.min_state_out = None
        self.max_state_out = None
        self.min_reward = None
        self.max_reward = None
        self.max_velocity = None
        self.acceleration = None
        self.damping_per_sec = None
        self.max_sensor_distance = None
        self.pre_train_episodes = None
        self.optimizer = None
        self.best_score = None
        self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                cfg = yaml.load(stream)['neural_network']
            except yaml.YAMLError as exc:
                print "didn't load"
                print(exc)

        self.max_velocity = cfg['dynamics']['max_velocity']
        self.speed_limit = cfg['dynamics']['speed_limit']
        self.brake_per_sec = cfg['dynamics']['brake_per_sec']
        self.acceleration = cfg['dynamics']['acceleration']
        self.damping_per_sec = cfg['dynamics']['damping_per_sec']
        self.max_sensor_distance = cfg['sensors']['max_sensor_distance']
        self.update_rate = cfg['controls']['update_rate']
        self.init_scale = cfg['network']['init_scale']
        self.init_constant = cfg['network']['init_constant']
        self.logging_freq = cfg['network']['logging_freq']
        self.max_learning_rate = cfg['network']['max_learning_rate']
        self.min_learning_rate = cfg['network']['min_learning_rate']
        self.learning_rate_time = cfg['network']['learning_rate_time']
        self.max_grad_norm = cfg['network']['max_grad_norm']
        self.trace_length = cfg['network']['trace_length']
        self.hidden_size_dense = cfg['network']['hidden_size_dense']
        self.hidden_size_rnn = cfg['network']['hidden_size_rnn']
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
        self.epsilon_start = cfg['network']['epsilon_start']
        self.epsilon_end = cfg['network']['epsilon_end']
        self.epsilon_time = cfg['network']['epsilon_time']
        self.epsilon = cfg['network']['epsilon']
        self.gamma = cfg['network']['gamma']
        self.store_every_ts = cfg['network']['store_every_ts']
        self.train_freq = cfg['network']['train_freq']
        self.clamp = cfg['network']['clamp']
        self.weight_folder = cfg['network']['weight_folder']
        self.log_folder = cfg['network']['log_folder']
        self.alpha = cfg['network']['alpha']
        self.beta = cfg['network']['beta']
        self.use_QDQN = cfg['network']['use_QDQN']
        self.use_double_q = cfg['network']['use_double_q']
        self.min_update_freq = cfg['network']['min_update_freq']
        self.max_update_freq = cfg['network']['max_update_freq']
        self.min_state_out = cfg['network']['min_state_out']
        self.max_state_out = cfg['network']['max_state_out']
        self.min_reward = cfg['network']['min_reward']
        self.max_reward = cfg['network']['max_reward']
        self.pre_train_episodes = cfg['network']['pre_train_episodes']
        self.optimizer = cfg['network']['optimizer']
        self.best_score = cfg['network']['best_score']
        self.max_epoch = cfg['network']['max_epoch']
        self.min_history = cfg['network']['min_history']
        self.update_length = cfg['network']['update_length']
        self.use_angle_reward = cfg['network']['use_angle_reward']
