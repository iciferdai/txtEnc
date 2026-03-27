import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d|%(levelname)s|%(filename)s:%(lineno)d|%(funcName)s -> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

DEFAULT_BATCH_SIZE = 8
D_MODEL = 1024
NUM_HEADS = 8
D_K = D_MODEL // NUM_HEADS
assert D_MODEL % NUM_HEADS == 0
HIDDEN_SIZE = D_MODEL * 4
DROPOUT_RATE = 0.1
MAX_LEN = 100
POS_ENCODING_BASE = 10000.0
ENC_LAYER_NUM = 3
FROZE_LAYER_NUM = 2
NEGATIVE_SAMPLE_NUM = 8

CLS_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3
SEP_ID = 4
MASK_ID = 5
UNK_ID = 6
REM_ID_1 = 7
REM_ID_2 = 8
REM_ID_3 = 9
CUS_START_ID = 10
IGNORE_ID = -100

TEST_PERCENT = 0.2

