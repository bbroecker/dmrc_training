# dmrc_training
Distance based multi robot coordination
# Test trained weights
Execute ``` ./main/test_both_nn.py ```
The weights are loaded frome the config ``./main/pretrained_weights/dqn/nn_config.py`` and ``./main/pretrained_weights/rnn/nn_config.py`` respectivly.

# Train DQN network
Execute ``` ./main/train_dqn.py ```. The weights are stored in ``./main/weight_save/dqn``. If you want to test the newly trained weights, change the ``weight_folder`` paramater in ``./main/pretrained_weights/dqn/nn_config.py`` to the newly created folder.

# Train RNN network
Execute ``` ./main/train_rnn.py ```. The weights are stored in ``./main/weight_save/rnn``. If you want to test the newly trained weights, change the ``weight_folder`` paramater in ``./main/pretrained_weights/rnn/nn_config.py`` to the newly created folder.
