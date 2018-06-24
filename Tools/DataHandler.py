import os

import pandas as pd


class DataHandler(object):
    def __init__(self, num_buckets):
        self.data_frame = None
        self.episode_buckets = [dict() for x in range(num_buckets)]
        self.merged_buffer = {}
        self.num_buckets = num_buckets
        self.reset_merge_buffer()
        self.reset_episode_buffer()


    # def store_dict(self, bucket_id, data):
    #     self.row += 1
    #     # self.data_buckets[bucket_id].append(data)
    #     data["episode"] = [self.episode_counter + bucket_id]
    #     if self.data_frame is None:
    #         self.data_frame = pd.DataFrame(data)
    #     else:
    #         for key in data.keys():
    #             self.data_frame.loc[self.row, key] = data[key][0]

    def store_data(self, key, data, bucket_id):
        self.episode_buckets[bucket_id][key].append(data)


    def reset_merge_buffer(self):
        self.merged_buffer = {}
        self.merged_buffer["episode"] = []
        self.merged_buffer["step_count"] = []
        self.merged_buffer["x_pos"] = []
        self.merged_buffer["y_pos"] = []
        self.merged_buffer["x_vel"] = []
        self.merged_buffer["y_vel"] = []
        self.merged_buffer["my_speed"] = []
        self.merged_buffer["my_angle"] = []
        self.merged_buffer["obstacle_local_speed"] = []
        self.merged_buffer["obstacle_local_angle"] = []
        self.merged_buffer["obstacle_distance"] = []
        self.merged_buffer["obstacle_angle"] = []
        self.merged_buffer["goal_distance"] = []
        self.merged_buffer["goal_angle"] = []
        self.merged_buffer["max_sensor_distance"] = []
        self.merged_buffer["max_velocity"] = []
        self.merged_buffer["update_rate"] = []
        self.merged_buffer["new_goal"] = []
        self.merged_buffer["new_obs"] = []

    def reset_episode_buffer(self):
        self.episode_buckets = [dict() for x in range(self.num_buckets)]
        for i in range(self.num_buckets):
            self.episode_buckets[i]["episode"] = []
            self.episode_buckets[i]["step_count"] = []
            self.episode_buckets[i]["x_pos"] = []
            self.episode_buckets[i]["y_pos"] = []
            self.episode_buckets[i]["x_vel"] = []
            self.episode_buckets[i]["y_vel"] = []
            self.episode_buckets[i]["my_speed"] = []
            self.episode_buckets[i]["my_angle"] = []
            self.episode_buckets[i]["obstacle_local_speed"] = []
            self.episode_buckets[i]["obstacle_local_angle"] = []
            self.episode_buckets[i]["obstacle_distance"] = []
            self.episode_buckets[i]["obstacle_angle"] = []
            self.episode_buckets[i]["goal_distance"] = []
            self.episode_buckets[i]["goal_angle"] = []
            self.episode_buckets[i]["max_sensor_distance"] = []
            self.episode_buckets[i]["max_velocity"] = []
            self.episode_buckets[i]["update_rate"] = []
            self.episode_buckets[i]["new_goal"] = []
            self.episode_buckets[i]["new_obs"] = []

    def merge_buffer(self):
        for i in range(self.num_buckets):
            for k in self.episode_buckets[i]:
                self.merged_buffer[k].extend(self.episode_buckets[i][k])
        self.reset_episode_buffer()

    def save_csv(self, file_name):
        data_frame = pd.DataFrame(self.merged_buffer)
        if os.path.exists(file_name):
            # with open(file_name, 'a') as f:
            data_frame.to_csv(file_name, mode='a', header=False, index=False, compression='gzip')
        else:
            data_frame.to_csv(file_name, index=False, compression='gzip')




