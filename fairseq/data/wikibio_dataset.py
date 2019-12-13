import numpy as np
import torch

from torch.utils.data import Dataset
from fairseq.data import data_utils, FairseqDataset



# padding操作是在这里做的
def collate(samples, pad_idx):
    if len(samples) == 0:
        return {}
    
    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = [s['source'] for s in samples] 
    src_lengths = [s['source'].size(0) for s in samples]
    max_src_length = max(src_lengths)

    src_lengths = torch.LongTensor(src_lengths) # batch
    src_data = torch.LongTensor(len(samples), max_src_length, 4).fill_(pad_idx)  # batch * src_len * 4
    for i, v in enumerate(src_tokens):
        src_data[i][:v.size(0)].copy_(v)

    # sort by descending source length 
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_data = src_data.index_select(0, sort_order)

    # target部分
    # target的text已经是 <bos> + text + <eos> 了
    # 这里需要给他造一个teaching force的prev_output_tokens，这个是把<eos>去了
    max_tgt_length = max([s['target'].size(0) for s in samples])
    tgt_tokens = [s['target'] for s in samples]
    tgt_data = torch.LongTensor(len(samples), max_tgt_length-1).fill_(pad_idx) # batch * tgt_len(+eos)
    for i, v in enumerate(tgt_tokens):
        tgt_data[i][:v.size(0)-1].copy_(v[1:])
    tgt_data = tgt_data.index_select(0, sort_order)
    ntokens = sum(len(s['target'])-2 for s in samples)  # 减去<bos> 和 <eos>

    prev_output_tokens = torch.LongTensor(len(samples), max_tgt_length-1).fill_(pad_idx) # batch * tgt_len(+bos)
    for i, v in enumerate(tgt_tokens):
        prev_output_tokens[i][:v.size(0)-1].copy_(v[:-1])
    prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

    # 注意
    # target 是 text + <eos>
    # prev_output_tokens 是 <bos> + text
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_data,
            'src_lengths': src_lengths,
        },
        'target': tgt_data,
    }

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


# WikiTableDataset 和 WikiTextDataset中已经truncate，但是没有bos和eos
# bos和eos在WikiBioDataset的getitem的时候才加上
# collate返回的结果中，target有bos和eos，prev_output_tokens没有eos有bos

class WikiBioDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        dict (~fairseq.data.Dictionary, optional): vocabulary
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        truncate (bool, optional): if surpass max positions, truncate it rather 
            than discard it (default: False)
    """

    def __init__(
        self, src, src_sizes, 
        tgt=None, tgt_sizes=None, word_dict=None,
        shuffle=True,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.dict = word_dict
        self.shuffle = shuffle

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None  
        src_item = self.src[index]

        # src和tgt dataset出来的已经是tensor了!

        # target端的加上bos和eos
        tgt_item = torch.cat((torch.LongTensor([self.dict.bos()]),self.tgt[index], torch.LongTensor([self.dict.eos()])))
        
        # src_item: src_length * 4
        # trg_item: trg_length

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.dict.pad())

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)




class WikiTableData(Dataset):
    '''
    用于生成table2text的数据集
    使用：
        变成word id的box.val
        变成field id的box.lab
        pos
        rpos
    '''
    def __init__(self, box_val_file, box_lab_file, pos_file, rpos_file, truncate_length=1024): 
        box_val = open(box_val_file, 'r').read().splitlines()
        box_lab = open(box_lab_file, 'r').read().splitlines()
        pos = open(pos_file, 'r').read().splitlines()
        rpos = open(rpos_file, 'r').read().splitlines()
        
        assert len(box_val) == len(box_lab)
        assert len(box_lab) == len(pos)
        assert len(pos) == len(rpos)

        self.data = [] # 里面是tensor
        self.src_size = []
        for i in range(len(box_val)):
            val = [int(i) for i in box_val[i].split()]
            lab = [int(i) for i in box_lab[i].split()]
            p = [int(i) for i in pos[i].split()]
            rp = [int(i) for i in rpos[i].split()]

            if len(val) >= truncate_length:
                val = val[:truncate_length]
                lab = lab[:truncate_length]
                p = p[:truncate_length]
                rp = rp[:truncate_length]
                
            assert len(val) == len(lab)
            assert len(lab) == len(p)
            assert len(p) == len(rp)
            
            self.src_size.append(len(val))
            table = list(zip(val, lab, p, rp))
            table = torch.LongTensor(table)  # src_len * 4
            self.data.append(table)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.data[idx]



class WikiTextData(Dataset):
    '''
    用于生成table2text的数据集
    使用：
        summary id
    '''
    def __init__(self, sentences_file, truncate_length=1024):
        sentences = open(sentences_file, 'r').read().splitlines()
        self.data = [] # 里面是Tensor
        self.tgt_size = []
        for i in range(len(sentences)):
            st = [int(i) for i in sentences[i].split()]
            if len(st) >= truncate_length:
                st = st[:truncate_length]
            self.tgt_size.append(len(st))
            sentence = torch.LongTensor(st)
            self.data.append(sentence)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.data[idx]



    