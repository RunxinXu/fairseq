#!/usr/bin/env python3 -u
import torch
import os 
import nltk

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        if not args.attention_copy:
            model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
            )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)
    
    # 放生成的句子
    hypo_sentences = []
    # 放测试集句子
    refer_sentences = []

    tgt_sentences_list = []
    tgt_sentences_path = os.path.join(args.data, args.gen_subset, '{}.summary'.format(args.gen_subset)) 
    with open(tgt_sentences_path) as f:
        for line in f.readlines():
            line = line.strip()
            tgt_sentences_list.append(line.split()) # 这个是因为原句子就已经分好词了 所以可以直接split

    if args.attention_copy:
        src_sentences_path = os.path.join(args.data, args.gen_subset, '{}.val'.format(args.gen_subset))
        src_sentences_list = []
        with open(src_sentences_path) as f:
            for line in f.readlines():
                line = line.strip()
                src_sentences_list.append(line.split())

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):

                hypo = hypos[i][0] 
                generate_sentence = hypo['tokens'].tolist() 
                generate_sentence = [tgt_dict[x] for x in generate_sentence[:-1]]

                if args.attention_copy:
                    attention = hypo['attention']  # src_len * tgt_len

                    for i in range(len(generate_sentence)):
                        if generate_sentence[i] == tgt_dict.unk_word:
                            attend_src_index = torch.argmax(attention[:,i])
                            generate_sentence[i] = src_sentences_list[sample_id][attend_src_index]

                hypo_sentences.append(generate_sentence)
                refer_sentence = tgt_sentences_list[sample_id] 
                refer_sentences.append([refer_sentence])

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})

    bleu = nltk.translate.bleu_score.corpus_bleu(refer_sentences, hypo_sentences)
    print('bleu:')
    print(bleu)

def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
