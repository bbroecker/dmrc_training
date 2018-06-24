import yaml


class SimulationConfig(object):
    def __init__(self, config_file):
        self.critical_radius = None
        self.goal_update_distance = None
        self.total_meter_x = None
        self.wall_size_x = None
        self.total_meter_y = None
        self.wall_size_y = None
        self.drone_radius = None
        self.goal_update_steps = None
        self.max_sensor_distance = None
        self.num_drones = None
        self.area_pixel_x = None
        self.debug_pixel_x = None
        self.area_pixel_y = None
        self.max_epLength = None
        self.debug_pixel_y = None
        self.load_file(config_file)

    def load_file(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                cfg = yaml.load(stream)['simulation']
            except yaml.YAMLError as exc:
                print(exc)

        self.critical_radius = cfg['world']['critical_radius']
        self.slow_down_max = cfg['world']['slow_down_max']
        self.slow_down_min = cfg['world']['slow_down_min']
        self.goal_update_distance = cfg['world']['goal_update_distance']
        self.total_meter_x = cfg['world']['total_meter_x']
        self.wall_size_x = cfg['world']['wall_size_x']
        self.total_meter_y = cfg['world']['total_meter_y']
        self.wall_size_y = cfg['world']['wall_size_y']
        self.span_x = cfg['world']['span_x']
        self.span_y = cfg['world']['span_y']
        self.speed_limit = cfg['world']['speed_limit']
        self.slow_down_radius_max = cfg['world']['slow_down_radius_max']
        self.slow_down_radius_min = cfg['world']['slow_down_radius_min']
        self.drone_radius = cfg['world']['drone_radius']
        self.goal_update_steps = cfg['world']['goal_update_steps']
        self.num_drones = cfg['world']['num_drones']
        self.max_epLength = cfg['world']['max_epLength']
        self.area_pixel_x = cfg['frontend']['area_pixel_x']
        self.debug_pixel_x = cfg['frontend']['debug_pixel_x']
        self.area_pixel_y = cfg['frontend']['area_pixel_y']
        self.debug_pixel_y = cfg['frontend']['debug_pixel_y']