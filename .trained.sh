
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

# # train: 86.963, valid: 60.6904 / 12.1585
# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -optim adam -learning_rate 0.001 -save_model multi30k_dnc -gpus 0

# # train: 76.3139, valid: 62.9493 / 10.0759
# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -optim adam -learning_rate 0.0001 -save_model multi30k_dnc_lr_0.0001_nodecay -start_decay_at 100 -epochs 50 -gpus 0

# # train: 75.9789, valid: 58.2611 / 13.1999
# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -optim adam -learning_rate 0.001 -save_model multi30k_dnc_lr_0.001_nodecay -epochs 50 -gpus 0

# # train: 71.7379, valid: 62.033 / 10.1482
# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -optim adam -learning_rate 0.0001 -save_model multi30k_dnc_lr_0.0001_nodecay -epochs 50 -gpus 0

# # train: 43.1993, valid: 43.6852 / 25.914
# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -optim adadelta -learning_rate 1.0 -save_model multi30k_dnc_adadelta_lr_1_nodecay -epochs 20 -gpus 0

# # WITHOUT ATTENTION

# # train: 94.4911, valid: 57.3803 / 32.4766
# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 20 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model test20x500_nodropout -epochs 20 -gpus 0

# # without dropout: train: 92.5832, valid: 55.1641 / 41.7716
# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 20 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model test20x500_nodropout -start_decay_at 3 -epochs 10 -gpus 0

# # without dropout: train: 88.0908, valid: 53.715, 53.6979
# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 20 -layers 1 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model dnc_layers_1_cells_20x500_nodropout -start_decay_at 3 -epochs 10 -gpus 0

# # 2-layer LSTM as one layer controller: train: 89.2508, valid: 54.2193 / 58.6577
# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 20 -layers 1 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model dnc_layers_1x2_cells_20x500_nodropout -start_decay_at 3 -epochs 10 -gpus 0

# # 2-layer LSTM as one layer controller: train: na - 73.6643, valid: 56.741 / 15.078
# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 20 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model dnc_layers_2x2_cells_20x500_nodropout -start_decay_at 3 -epochs 10 -gpus 0


# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 2 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model dnc_lr_0.001_cells_2x500 -epochs 20 -gpus 0

# # With DNC v0.0.4
# python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 20 -cell_size 500 -read_heads 2 -optim adam -learning_rate 0.001 -save_model dncv0.0.4_layers_2x2_cells_20x500 -epochs 20 -gpus 0


# 62.8885 / valid: 58.3085 / 14.089
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adam -learning_rate 0.0001 -start_decay_at 15 -save_model dncv0.0.6_cells_50x100 -epochs 50 -gpus 0

# 76.1191 / valid: 62.0973 / 11.3639
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adam -learning_rate 0.0001 -start_decay_at 20 -save_model dncv0.0.6_cells_50x100 -epochs 50 -gpus 0

# 86.3343 / valid: 62.1825 / 10.7382
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adam -learning_rate 0.0001 -start_decay_at 40 -save_model dncv0.0.6_cells_50x100_decay_1t_40 -epochs 50 -gpus 0

# rmsprop & dropout 73.2124 valid: 62.8849 / 10.3056 / 20
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -learning_rate 0.0001 -optim rmsprop -start_decay_at 40 -save_model dncv0.0.6_cells_50x100_decay_1t_40-rmsprop -epochs 50 -gpus 0

# read heads 16 84.68 / valid: 61.4162 / 11.8242 / 35
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 16 -learning_rate 0.0001 -optim rmsprop -start_decay_at 40 -save_model dncv0.0.6_cells_50x100_decay_1t_40-rmsprop-read_heads-16 -epochs 50 -gpus 0

# SGD diverged
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 4 -learning_rate 1 -optim sgd -start_decay_at 20 -save_model dncv0.0.6_cells_50x100_decay_1t_40-read_heads-4 -epochs 25 -gpus 0

# adadelta 62.909 / valid: 60.6003 / 9.86059 / 8
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 4 -learning_rate 1 -optim adadelta -start_decay_at 20 -save_model dncv0.0.6_cells_50x100_decay_1t_40-read_heads-4 -epochs 25 -gpus 0

# adadelta 0.1 61.8746 / valid: 59.1245 / 12.9613
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 4 -learning_rate 0.1 -optim adadelta -start_decay_at 10 -save_model dncv0.0.6_cells_50x100_decay_1t_10-read_heads-4 -epochs 25 -gpus 0

# 72.6718 / valid: 61.175 / 11.2483
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 10 -cell_size 512 -read_heads 4 -learning_rate 0.0001 -optim adam -start_decay_at 15 -save_model dncv0.0.6_cells_10x512_decay_1t_15-read_heads-4 -epochs 25 -gpus 0

###########################
# no attn
###########################

# no attention 60.335 / valid: 52.1711 / 18.8026
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adam -learning_rate 0.0001 -start_decay_at 20 -save_model test -epochs 50 -gpus 0

# no attn no controller state 62.299 / valid: 56.8185 / 14.2657
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adam -learning_rate 0.0001 -start_decay_at 20 -save_model test -epochs 50 -gpus 0

# no attn no memory state 62.8132 / valid: 52.5827 / 18.7234
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adam -learning_rate 0.0001 -start_decay_at 30 -save_model test -epochs 50 -gpus 0

# no attn no controller state 2 layers 69.8895 / valid: 58.6349 / 13.8991
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adam -learning_rate 0.0001 -start_decay_at 30 -save_model test -epochs 50 -gpus 0

# no attn no controller state 2x2 layers, adadelta, independent_linears - 95.8026 / valid: 53.9095 / 17.822
python train.py -data data/multi30k.atok.low.train.pt -batch_size 50 -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adadelta -learning_rate 0.5 -start_decay_at 30 -save_model test -epochs 50 -gpus 0

# 4 read heads 78.2539 / valid: 59.018 / 15.7507
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 4 -optim adam -learning_rate 0.0001 -start_decay_at 30 -batch_size 25 -save_model test -epochs 50 -gpus 0

# more memory 92.4379 / valid: 57.7622 / 24.3431
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 200 -cell_size 50 -read_heads 4 -optim adam -learning_rate 0.0005 -start_decay_at 10 -batch_size 25 -save_model 4readheads -epochs 30 -gpus 0

# 8 read heads 91.9909 / valid: 56.1586 / 20.5365
python train.py -data data/multi30k.atok.low.train.pt -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 8 -optim adam -learning_rate 0.001 -save_model 8readheads -epochs 25 -gpus 0

# baseline no attention 91.1548 / valid: 58.046 / 21.3743
python train.py -data data/multi30k.atok.low.train.pt -start_decay_at 30 -epochs 50 -save_model multi30k_baseline_no_attn -gpus 0

# baseline no attn adam 63.3249 / valid: 53.2496 / 18.645
python train.py -data data/multi30k.atok.low.train.pt -start_decay_at 30 -epochs 50 -save_model multi30k_baseline_no_attn -gpus 0 -optim adam -learning_rate 0.0001

# baseline no attn adamax 59.331 / valid: 53.4412 / 19.4002
python train.py -data data/multi30k.atok.low.train.pt -start_decay_at 30 -epochs 50 -save_model multi30k_baseline_no_attn -gpus 0 -optim adamax -learning_rate 0.0002

# baseline no attn adagrad 46.5104 / valid: 44.6218 / 28.8539
python train.py -data data/multi30k.atok.low.train.pt -epochs 20 -save_model multi30k_baseline_no_attn -gpus 0 -optim adagrad -learning_rate 0.1


############################
# After removing stale files, with attention
############################

# 39.1869 / 31.653
python train.py -data data/multi30k.atok.low.train.pt -batch_size 100 -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim sgd -learning_rate 0.1 -start_decay_at 10 -save_model test -epochs 50 -gpus 0

# 73.6105 / valid: 65.7585 / 7.61542
python train.py -data data/multi30k.atok.low.train.pt -batch_size 100 -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adadelta -learning_rate 1 -start_decay_at 3 -save_model test -epochs 50 -gpus 0

# 4 read heads 73.1495 / valid: 65.8862 / 7.5111
python train.py -data data/multi30k.atok.low.train.pt -batch_size 100 -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 4 -optim adadelta -learning_rate 1 -start_decay_at 5 -save_model test -epochs 15 -gpus 0

# bigger net (account for read vectors) 77.1077 / valid: 66.5744 / 7.26448
python train.py -rnn_size 700 -word_vec_size 700 -data data/multi30k.atok.low.train.pt -batch_size 90 -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adadelta -learning_rate 1 -start_decay_at 5 -save_model test -epochs 15 -gpus 0

# cell sizes 100
python train.py -data data/multi30k.atok.low.train.pt -batch_size 100 -rnn_type DNC -nr_cells 100 -cell_size 100 -read_heads 2 -optim adadelta -learning_rate 1 -start_decay_at 5 -save_model test -epochs 15 -gpus 0

# no pass through memory - 71.6292  /
python train.py -data data/multi30k.atok.low.train.pt -batch_size 100 -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adadelta -learning_rate 1 -start_decay_at 3 -save_model test -epochs 50 -gpus 0


# WMT

python train.py -dropout 0.5 -data data/wmt.atok.low.train.pt -batch_size 50 -rnn_type DNC -nr_cells 100 -cell_size 50 -read_heads 2 -optim adadelta -learning_rate 1 -start_decay_at 5 -save_model test -epochs 15 -gpus 1

# ~ 50 1st epoch
python train.py -data data/wmt.atok.low.train.pt -save_model wmt_baseline -gpus 0 -optim adadelta -learning_rate 1

