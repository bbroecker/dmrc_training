import yaml


class Environment(object):

    def __init__(self, config_file):
        pass
        # self.num_episodes = None
        # self.pre_train_steps = None
        # self.max_epLength = None
        # self.load_model = None
        # self.batch_freq = None
        # self.training_freq = None
        # self.h_size = None
        # self.save_ep_rate = None
        # self.gamma = None
        # self.startE = None
        # self.endE = None
        # self.trace_length = None
        # self.anneling_steps = None
        # self.batch_size = None
        # self.num_layers = None
        # self.load_basic_config(config_file)

    # def load_basic_config(self, config_file):
    #     with open(config_file, 'r') as stream:
    #         try:
    #             cfg = yaml.load(stream)['Environment']
    #         except yaml.YAMLError as exc:
    #             print(exc)
    #
    #     self.num_episodes = cfg['network']['num_episodes']
    #     self.pre_train_steps = cfg['network']['pre_train_steps']
    #     self.max_epLength = cfg['network']['max_epLength']
    #     self.load_model = cfg['network']['load_model']
    #     self.batch_freq = cfg['network']['batch_freq']
    #     self.training_freq = cfg['network']['training_freq']
    #     self.h_size = cfg['network']['h_size']
    #     self.save_ep_rate = cfg['network']['save_ep_rate']
    #     self.h_size = cfg['network']['h_size']
    #     self.batch_size = cfg['network']['batch_size']
    #     self.gamma = cfg['network']['gamma']
    #     self.startE = cfg['network']['startE']
    #     self.endE = cfg['network']['endE']
    #     self.anneling_steps = cfg['network']['anneling_steps']
    #     self.trace_length = cfg['network']['trace_length']
    #     self.anneling_steps = cfg['network']['anneling_steps']
    #     self.num_layers = cfg['network']['num_layers']


    def reset(self):
        raise NotImplementedError("reset not implemented")

    def get_num_states(self):
        raise NotImplementedError("get_num_states not implemented")

    def get_num_actions(self):
        raise NotImplementedError("get_num_actions not implemented")

    def get_state(self):
        raise NotImplementedError("get_state not implemented")

    def sample_next_state(self, action):
        raise NotImplementedError("sample_next_state not implemented")

    def render(self):
        raise NotImplementedError("render not implemented")