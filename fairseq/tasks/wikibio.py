import itertools
import os
import sys
from fairseq import options, utils
from fairseq.data import (
    WikiBioDataset,
    WikiTableData,
    WikiTextData,
)

from fairseq.tasks import FairseqTask, register_task


def load_wikibio_dataset(
    data_path, split, word_dict, truncate_src_length, truncate_tgt_length):

    table = WikiTableData('{}/{}/{}.val.id'.format(data_path, split, split),
                    '{}/{}/{}.lab.id'.format(data_path, split, split),
                    '{}/{}/{}.pos'.format(data_path, split, split),
                    '{}/{}/{}.rpos'.format(data_path, split, split), truncate_src_length)
    text = WikiTextData('{}/{}/{}.summary.id'.format(data_path, split, split), truncate_tgt_length)
    return WikiBioDataset(table, table.src_size, text, text.tgt_size, word_dict)


@register_task('wikibio')
class WikibioTask(FairseqTask):
    """
    table to text

    Args:
        dict (~fairseq.data.Dictionary): dictionary for the source 
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--truncate-src-length', default=60, type=int, metavar='N',
                            help='truncate length of source')
        parser.add_argument('--truncate-tgt-length', default=60, type=int, metavar='N',
                            help='truncate length of target')
        # default不是50 因为编码的时候是1～50..
        parser.add_argument('--position-num', default=51, type=int, metavar='N',
                            help='position num')
        # fmt: on

    def __init__(self, args, word_dict, field_dict):
        super().__init__(args)
        self.word_dict = word_dict
        self.field_dict = field_dict
        self.pos_num = args.position_num

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        path = args.data # data 的目录 下有 train/valid/test目录
        assert path is not None

        # load dictionaries
        word_dict = cls.load_dictionary(os.path.join(path, 'word_vocab.txt'))
        field_dict = cls.load_dictionary(os.path.join(path, 'field_vocab.txt'))
        return cls(args, word_dict, field_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        path = self.args.data
        assert path is not None

        self.datasets[split] = load_wikibio_dataset(
            path, split, self.word_dict,
            truncate_src_length=self.args.truncate_src_length,
            truncate_tgt_length=self.args.truncate_tgt_length,
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.truncate_src_length, self.args.truncate_tgt_length)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.word_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.word_dict

    @property
    def field_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.field_dict
    
    @property
    def position_num(self):
        return self.pos_num
