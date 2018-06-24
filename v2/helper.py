import os
import numpy as np

def generate_path(src, cfg):
    output = src
    output += "lr" + str(cfg.max_learning_rate) + "_"
    output += "rnn_"
    for h in range(len(cfg.hidden_size_rnn)):
        if cfg.use_tanh_rnn[h]:
            output += "tanh" + str(cfg.hidden_size_rnn[h]) + "_"
        else:
            output += "relu" + str(cfg.hidden_size_rnn[h]) + "_"
    output += "dense_"
    for h in range(len(cfg.hidden_size_dense)):
        if cfg.use_tanh_dense[h]:
            output += "tanh" + str(cfg.hidden_size_dense[h]) + "_"
        else:
            output += "relu" + str(cfg.hidden_size_dense[h]) + "_"
    output += "batch" + str(cfg.batch_size) + "_"
    output += "trace_length" + str(cfg.trace_length) + "_"
    output += "update_length" + str(cfg.update_length) + "_"
    output += "forget_bias" + str(cfg.forget_bias) + "_"
    output += "init_value" + str(cfg.init_scale) + "_"
    output += "keep_prob" + str(cfg.keep_prob) + "_"
    output += "buffer_size" + str(cfg.buffer_size) + "_"
    # output += "num_layers" + str(cfg.num_layers) + "_"
    output += cfg.optimizer
    try:
        output += "_use_angle_reward" + str(cfg.use_angle_reward) + "_"
    except AttributeError:
        print "doesn't have _use_angle_reward"
    print output
    return output


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def create_nn_folders(cfg, weight_path, log_path, mkdir = True):
    load = False
    gen_weight_path = generate_path(weight_path, cfg)
    i = 0
    while os.path.exists(gen_weight_path + "_" + str(i)):
        i += 1
    gen_weight_path += "_" + str(i)
    if cfg.weight_folder is None or cfg.weight_folder == "None":
        cfg.weight_folder = gen_weight_path
    else:
        load = True

    gen_log_path = generate_path(log_path, cfg)
    i = 0
    while os.path.exists(gen_log_path + "_" + str(i)):
        i += 1
    gen_log_path += "_" + str(i)
    if cfg.log_folder is None or cfg.log_folder == "None":
        cfg.log_folder = gen_log_path

    # Make a path for our model to be saved in.
    if mkdir and not os.path.exists(gen_log_path):
        os.makedirs(gen_log_path)
    if mkdir and not os.path.exists(gen_weight_path):
        os.makedirs(gen_weight_path)

    # cfg.weight_folder = gen_weight_path
    # cfg.log_folder = gen_log_path

    return gen_weight_path, gen_log_path, load


def discrete_to_angle(discrete_angle, cfg):
    range_size = 2 * np.pi
    step_size = range_size / float(cfg.output_size - 1.0)
    return step_size * discrete_angle - np.pi


def angle_to_discrete(angle, cfg):
    range_size = 2 * np.pi
    step_size = range_size / float(cfg.output_size - 1.0)
    index = int(round((angle + np.pi) / step_size))
    return index

def wrap_angle(angle):
    a = angle
    while a > np.pi:
        a -= 2 * np.pi
    while a < -np.pi:
        a += 2 * np.pi
    return a

def kalman_xy(x, P, measurement, R,
              motion = np.matrix('0. 0. 0. 0.').T,
              Q = np.matrix(np.eye(4)),
              H = np.matrix('''1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      ''')):
    """
    Parameters:
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    return kalman(x, P, measurement, R, motion, Q,
                  F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''), H=H)

def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H
    '''
    # PREDICT x, P based on motion
    # NEW ORDER
    x = F*x + motion
    P = F*P*F.T + Q

    # UPDATE x, P based on measurement m
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    # print "{} \n".format(P)
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    # P = (I - K*H)*P #OG
    P = P - K*H*P

    # PREDICT x, P based on motion
    # OLD ORDER
    # x = F*x + motion
    # P = F*P*F.T + Q


    return x, P