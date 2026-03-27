from data_dict import *
#from data_dict_addition import *
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
import re
import random

def generate_src_mask(src_token_ids, pad_id = PAD_ID):
    src_mask = (src_token_ids == pad_id)
    # [batch_size, 1, 1, src_seq_len] -> [batch_size, n_heads, seq_len_q, seq_len_k]
    src_mask_4d = src_mask.unsqueeze(1).unsqueeze(1)
    logging.debug(f"Mask: {src_mask_4d.shape} -> {src_mask_4d.device}")
    return src_mask_4d


def generate_tgt_mask(tgt_token_ids, pad_id = PAD_ID):
    # [batch_size, tgt_seq_len]
    batch_size, tgt_seq_len = tgt_token_ids.shape

    # ahead mask -> [batch_size, tgt_seq_len, tgt_seq_len]
    ahead_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.bool), diagonal=1)
    ahead_mask = ahead_mask.unsqueeze(0).repeat(batch_size, 1, 1)

    # pad mask -> [batch_size, tgt_seq_len, tgt_seq_len]
    tgt_pad_mask = (tgt_token_ids == pad_id)
    tgt_pad_mask_3d = tgt_pad_mask.unsqueeze(1).repeat(1, tgt_seq_len, 1)

    # combine
    tgt_mask = ahead_mask | tgt_pad_mask_3d
    # [batch_size, 1, tgt_seq_len, tgt_seq_len] -> [batch_size, n_heads, seq_len_q, seq_len_k]
    tgt_mask_4d = tgt_mask.unsqueeze(1)
    logging.debug(f"Mask: {tgt_mask_4d.shape} -> {tgt_mask_4d.device}")
    return tgt_mask_4d


def generate_mlm_mask(input_ids):
    masked_input = input_ids.clone()
    mlm_labels = torch.full_like(input_ids, IGNORE_ID)
    maskable = (input_ids != CLS_ID) & (input_ids != PAD_ID)
    mask = torch.rand_like(input_ids.float()) < 0.15
    mask = mask & maskable

    for i, j in zip(*torch.nonzero(mask, as_tuple=True)):
        rand = torch.rand(1).item()
        if rand < 0.8:
            masked_input[i, j] = MASK_ID
        elif rand < 0.9:
            masked_input[i, j] = torch.randint(CUS_START_ID, VOCAB_SIZE, (1,)).item()
        mlm_labels[i, j] = input_ids[i, j]

    return masked_input, mlm_labels

def cov_ids(txt):
    ids = [CLS_ID]
    for t in txt:
        ids.append(token2idx[t])
    left_num = MAX_LEN - len(ids)
    ids += [PAD_ID] * left_num
    return ids


def process_data():
    src_ids=[]
    for msg, _ in demo_data:
        src_ids.append(cov_ids(msg))

    src_tensor = torch.tensor(src_ids, dtype=torch.long)
    dataset = TensorDataset(src_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        drop_last=False)
    return dataloader

def process_sft_data():
    positive_list=[]
    negative_list=[]
    normal_list=[]
    for msg, flag in demo_data:
        if flag == 1:
            positive_list.append(cov_ids(msg))
        elif flag == -1:
            negative_list.append(cov_ids(msg))
        else:
            normal_list.append(cov_ids(msg))

    #print(f'len: {len(positive_list)}|{len(negative_list)}|{len(normal_list)}')
    sft_data = positive_list, negative_list, normal_list
    return sft_data

if __name__ == '__main__':
    """
    d = process_data()
    for i, t in enumerate(d):
        print(f'{i}: {len(t)}|{t[0].shape}')
        src, tgt = generate_mlm_mask(t[0])
        print(src[0])
        print(tgt[0])
        break
    """
    d = process_sft_data()
    print(d)