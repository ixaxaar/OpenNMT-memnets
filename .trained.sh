
# baseline 1, train: 78.4124, valid: 69.7684 / 6.03863
python train.py -data data/multi30k.atok.low.train.pt -save_model multi30k_baseline -gpus 0

# baseline 2, adam no beta, eps, train: 74.5969, valid: 61.0953 / 11.5747
python train.py -data data/multi30k.atok.low.train.pt -rnn_type LSTM -optim adam -learning_rate 0.0001 -save_model multi30k_lstm_lr_0.0001_nodecay -epochs 50 -gpus 0

# baseline 3, train: 81.2668, valid: 67.9642 / 8.26244
python train.py -data data/multi30k.atok.low.train.pt -rnn_type LSTM -optim sgd -learning_rate 1.0 -save_model multi30k_lstm_lr_1_sgd_nodecay -epochs 50 -gpus 0

# baseline 4, train: 84.0381, valid: 68.3691 / 8.29581
python train.py -data data/multi30k.atok.low.train.pt -rnn_type LSTM -optim sgd -learning_rate 1.0 -save_model multi30k_lstm_lr_1_sgd_nodecay -epochs 50 -gpus 0 -input_feed 1

# baseline 5, train: 65.6955, valid: 61.6423 / 10.6165
python train.py -data data/multi30k.atok.low.train.pt -rnn_type LSTM -optim adam -learning_rate 0.0001 -save_model multi30k_lstm_lr_0.0001_nodecay -epochs 50 -gpus 0 -input_feed 1

# baseline 6, without attention
python train.py -data data/multi30k.atok.low.train.pt -rnn_type LSTM -optim adam -learning_rate 0.001 -save_model multi30k_lstm_lr_0.001_no_attn -epochs 10 -start_decay_at 3 -gpus 0

# train: 86.963, valid: 60.6904 / 12.1585
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -optim adam -learning_rate 0.001 -save_model multi30k_dnc -gpus 0

# train: 76.3139, valid: 62.9493 / 10.0759
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -optim adam -learning_rate 0.0001 -save_model multi30k_dnc_lr_0.0001_nodecay -start_decay_at 100 -epochs 50 -gpus 0

# train: 75.9789, valid: 58.2611 / 13.1999
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -optim adam -learning_rate 0.001 -save_model multi30k_dnc_lr_0.001_nodecay -epochs 50 -gpus 0

# train: 71.7379, valid: 62.033 / 10.1482
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -optim adam -learning_rate 0.0001 -save_model multi30k_dnc_lr_0.0001_nodecay -epochs 50 -gpus 0

# train: 43.1993, valid: 43.6852 / 25.914
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -optim adadelta -learning_rate 1.0 -save_model multi30k_dnc_adadelta_lr_1_nodecay -epochs 20 -gpus 0

# WITHOUT ATTENTION

# train: 94.4911, valid: 57.3803 / 32.4766
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 20 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model test20x500_nodropout -epochs 20 -gpus 0

# without dropout: train: 92.5832, valid: 55.1641 / 41.7716
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 20 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model test20x500_nodropout -start_decay_at 3 -epochs 10 -gpus 0

# without dropout: train: 88.0908, valid: 53.715, 53.6979
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 20 -layers 1 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model dnc_layers_1_cells_20x500_nodropout -start_decay_at 3 -epochs 10 -gpus 0

# 2-layer LSTM as one layer controller: train: 89.2508, valid: 54.2193 / 58.6577
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 20 -layers 1 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model dnc_layers_1x2_cells_20x500_nodropout -start_decay_at 3 -epochs 10 -gpus 0

# 2-layer LSTM as one layer controller: train: na - 73.6643, valid: 56.741 / 15.078
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 20 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model dnc_layers_2x2_cells_20x500_nodropout -start_decay_at 3 -epochs 10 -gpus 0


python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 2 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model dnc_lr_0.001_cells_2x500 -epochs 20 -gpus 0
