import torch
import typing as tp


def stacking_collate_fn(batch: tp.Dict[str, tp.Any]):
    "batch = {'input': tensor, 'idx': ints, 'w': ints, 'gt_text': long tensor, 'tgt_len': long tensor}"
    result = {}
    result['input'] = torch.cat([b['input'] for b in batch], dim=0)
    idxs = []
    for im_dict in batch:
        num_chunks = im_dict['input'].size(0)
        idxs.extend([im_dict['idx']] * num_chunks)
    result['idx'] = torch.LongTensor(idxs)
    result['w'] = torch.LongTensor([b['w'] for b in batch])
    result['gt_text'] = torch.nn.utils.rnn.pad_sequence([b['gt_text'] for b in batch], batch_first=True)
    result['tgt_len'] = torch.LongTensor([b['tgt_len'] for b in batch])
    return result
