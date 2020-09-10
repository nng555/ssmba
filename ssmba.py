import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead

def hf_masked_encode(
        tokenizer,
        sentence: str,
        *addl_sentences,
        mask_prob=0.0,
        random_token_prob=0.0,
        leave_unmasked_prob=0.0):

    if random_token_prob > 0.0:
        weights = np.ones(len(tokenizer.vocab))
        weights[tokenizer.all_special_ids] = 0
        for k, v in tokenizer.vocab.items():
            if '[unused' in k:
                weights[v] = 0
        weights = weights / weights.sum()

    tokens = np.asarray(tokenizer.encode(sentence, *addl_sentences, add_special_tokens=True))

    if mask_prob == 0.0:
        return tokens

    sz = len(tokens)
    mask = np.full(sz, False)
    num_mask = int(mask_prob * sz + np.random.rand())

    mask_choice_p = np.ones(sz)
    for i in range(sz):
        if tokens[i] in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id]:
            mask_choice_p[i] = 0
    mask_choice_p = mask_choice_p / mask_choice_p.sum()

    mask[np.random.choice(sz, num_mask, replace=False, p=mask_choice_p)] = True

    mask_targets = np.full(len(mask), tokenizer.pad_token_id)
    mask_targets[mask] = tokens[mask == 1]

    # decide unmasking and random replacement
    rand_or_unmask_prob = random_token_prob + leave_unmasked_prob
    if rand_or_unmask_prob > 0.0:
        rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
        if random_token_prob == 0.0:
            unmask = rand_or_unmask
            rand_mask = None
        elif leave_unmasked_prob == 0.0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
            decision = np.random.rand(sz) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)
    else:
        unmask = rand_mask = None

    if unmask is not None:
        mask = mask ^ unmask

    tokens[mask] = tokenizer.mask_token_id
    if rand_mask is not None:
        num_rand = rand_mask.sum()
        if num_rand > 0:
            tokens[rand_mask] = np.random.choice(
                len(tokenizer.vocab),
                num_rand,
                p=weights,
            )

    return torch.tensor(tokens).long(), torch.tensor(mask).long()

def hf_reconstruction_prob_tok(masked_tokens, target_tokens, tokenizer, model, softmax_mask, reconstruct=False, topk=1):
    single = False

    # expand batch size 1
    if masked_tokens.dim() == 1:
        single = True
        masked_tokens = masked_tokens.unsqueeze(0)
        target_tokens = target_tokens.unsqueeze(0)

    masked_fill = torch.ones_like(masked_tokens)

    masked_index = (target_tokens != tokenizer.pad_token_id).nonzero(as_tuple=True)
    masked_orig_index = target_tokens[masked_index]

    # edge case of no masked tokens
    if len(masked_orig_index) == 0:
        if reconstruct:
            return masked_tokens, masked_fill
        else:
            return 1.0

    masked_orig_enum = [list(range(len(masked_orig_index))), masked_orig_index]

    outputs = model(
        masked_tokens.long().to(device=next(model.parameters()).device),
        masked_lm_labels=target_tokens
    )

    features = outputs[1]

    logits = features[masked_index]
    for l in logits:
        l[softmax_mask] = float('-inf')
    probs = logits.softmax(dim=-1)


    if (reconstruct):

        # sample from topk
        if topk != -1:
            values, indices = probs.topk(k=topk, dim=-1)
            kprobs = values.softmax(dim=-1)
            if (len(masked_index) > 1):
                samples = torch.cat([idx[torch.multinomial(kprob, 1)] for kprob, idx in zip(kprobs, indices)])
            else:
                samples = indices[torch.multinomial(kprobs, 1)]

        # unrestricted sampling
        else:
            if (len(masked_index) > 1):
                samples = torch.cat([torch.multinomial(prob, 1) for prob in probs])
            else:
                samples = torch.multinomial(probs, 1)

        # set samples
        masked_tokens[masked_index] = samples
        masked_fill[masked_index] = samples

        if single:
            return masked_tokens[0], masked_fill[0]
        else:
            return masked_tokens, masked_fill

    return torch.sum(torch.log(probs[masked_orig_enum])).item()

def fill_batch(args,
               tokenizer,
               sents,
               l,
               lines,
               labels,
               next_sent,
               num_gen,
               num_tries,
               gen_index):

    # load sentences into batch until full
    while(len(sents) < args.batch):

        print("###########")
        print(sents)
        # search for the next valid sentence
        while True:
            if next_sent >= len(lines[0]):
                break

            next_sents = [s_list[next_sent][0] for s_list in lines]
            next_len = len(tokenizer.encode(*next_sents))

            # skip input if too short or long
            if next_len > args.min_len and next_len < args.max_len:
                break
            next_sent += 1

        print(next_sent)

        # add it to our lists
        if next_sent < len(lines[0]):
            next_sent_lists = [s_list[next_sent] for s_list in lines]
            sents.append(list(zip(*next_sent_lists)))
            l.append(labels[next_sent])

            num_gen.append(0)
            num_tries.append(0)
            gen_index.append(0)
            next_sent += 1
        else:
            break

        print("#########")

    return sents, l, next_sent, num_gen, num_tries, gen_index



def gen_neighborhood(args):

    # initialize seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # load model and tokenizer
    r_model = AutoModelWithLMHead.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    r_model.eval()
    if torch.cuda.is_available():
        r_model.cuda()

    # remove unused vocab and special ids from sampling
    softmax_mask = np.full(len(tokenizer.vocab), False)
    softmax_mask[tokenizer.all_special_ids] = True
    for k, v in tokenizer.vocab.items():
        if '[unused' in k:
            softmax_mask[v] = True

    # load the inputs and labels
    lines = [tuple(s.strip().split('\t')) for s in open(args.in_file).readlines()]
    num_lines = len(lines)
    lines = [[[s] for s in s_list] for s_list in list(zip(*lines))]

    labels = [s.strip() for s in open(args.label_file).readlines()]

    # shard the input and labels
    if args.num_shards > 0:
        shard_start = (int(num_lines/args.num_shards) + 1) * args.shard
        shard_end = (int(num_lines/args.num_shards) + 1) * (args.shard + 1)
        lines = [s_list[shard_start:shard_end] for s_list in lines]
        labels = labels[shard_start:shard_end]

    # open output files
    s_rec_file = open(args.output_prefix + '_' + str(args.shard), 'w')
    l_rec_file = open(args.output_prefix + '.label', 'w')

    # sentences and labels to process
    sents = []
    l = []

    # number sentences generated
    num_gen = []

    # sentence index to noise from
    gen_index = []

    # number of tries generating a new sentence
    num_tries = []

    # next sentence index to draw from
    next_sent = 0

    sents, l, next_sent, num_gen, num_tries, gen_index = \
            fill_batch(args,
                       tokenizer,
                       sents,
                       l,
                       lines,
                       labels,
                       next_sent,
                       num_gen,
                       num_tries,
                       gen_index)

    # main augmentation loop
    while (sents != []):
        print(sents)

        # remove any sentences that are done generating and dump to file
        for i in range(len(num_gen))[::-1]:
            if num_gen[i] == args.num_samples or num_tries[i] > args.max_tries:

                # get sent info
                gen_sents = sents.pop(i)
                num_gen.pop(i)
                gen_index.pop(i)
                label = l.pop(i)

                # write generated sentences
                for sg in gen_sents[1:]:
                    s_rec_file.write('\t'.join([repr(val)[1:-1] for val in sg]) + '\n')
                    l_rec_file.write(label + '\n')

        # fill batch
        sents, l, next_sent, num_gen, num_tries, gen_index = \
                fill_batch(args,
                           tokenizer,
                           sents,
                           l,
                           lines,
                           labels,
                           next_sent,
                           num_gen,
                           num_tries,
                           gen_index)

        # break if done dumping
        if len(sents) == 0:
            break

        # build batch
        toks = []
        masks = []

        for i in range(len(gen_index)):
            s = sents[i][gen_index[i]]
            tok, mask = hf_masked_encode(
                    tokenizer,
                    *s,
                    mask_prob=args.mask_prob,
                    random_token_prob=args.random_token_prob,
                    leave_unmasked_prob=args.leave_unmasked_prob,
            )
            toks.append(tok)
            masks.append(mask)

        # pad up to max len input
        max_len = max([len(tok) for tok in toks])
        pad_tok = tokenizer.pad_token_id

        toks = [F.pad(tok, (0, max_len - len(tok)), 'constant', pad_tok) for tok in toks]
        masks = [F.pad(mask, (0, max_len - len(mask)), 'constant', pad_tok) for mask in masks]
        toks = torch.stack(toks)
        masks = torch.stack(masks)

        # load to GPU if available
        if torch.cuda.is_available():
            toks = toks.cuda()
            masks = masks.cuda()

        # predict reconstruction
        rec, rec_masks = hf_reconstruction_prob_tok(toks, masks, tokenizer, r_model, softmax_mask, reconstruct=True, topk=args.topk)

        # decode reconstructions and append to lists
        for i in range(len(rec)):
            rec_work = rec[i].cpu().tolist()
            s_rec = [s.strip() for s in tokenizer.decode([val for val in rec_work if val != tokenizer.pad_token_id][1:-1]).split(tokenizer.sep_token)]
            s_rec = tuple(s_rec)

            # check if identical reconstruction or empty
            if s_rec not in sents[i] and '' not in s_rec:
                sents[i].append(s_rec)
                num_gen[i] += 1
                num_tries[i] = 0
                gen_index[i] = 0

            # otherwise try next sentence
            else:
                num_tries[i] += 1
                gen_index[i] += 1
                if gen_index[i] == len(sents[i]):
                    gen_index[i] = 0

        # clean up tensors
        del toks
        del masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', '-s', type=int, default=0,
            help='Shard of input to process. Output filename '
            'will have _${shard} appended.')

    parser.add_argument('--num-shards', type=int, default=1,
            help='Total number of shards to shard input file with.')

    parser.add_argument('--seed', type=int,
            help='Random seed to use for reconstruction and noising.')

    parser.add_argument('--model', '-m', type=str,
            help='Name of HuggingFace BERT model to use for reconstruction,'
            ' or filepath to local model directory.')

    parser.add_argument('--tokenizer', type=str, default=None,
            help='Name of HuggingFace tokenizer to use for vocabulary'
            ' or filepath to local tokenizer. If None, uses the same'
            ' as model.')

    parser.add_argument('--in-file', '-i', type=str,
            help='Path of input text file for augmentation.'
            ' Inputs should be separated by newlines with tabs indicating'
            ' BERT <SEP> tokens.')

    parser.add_argument('--label-file', '-l', type=str,
            help='Path of input label file for augmentation if using '
            ' label preservation.' )

    parser.add_argument('--output-prefix', '-o', type=str,
            help='Prefix path for output files, including augmentations and'
            ' preserved labels.')

    parser.add_argument('--mask-prob', '-p', type=float, default=0.15,
            help='Probability for selecting a token for noising.'
            ' Selected tokens are then masked, randomly replaced,'
            ' or left the same.')

    parser.add_argument('--random-token-prob', '-r', type=float, default=0.1,
            help='Probability of a selected token being replaced'
            ' randomly from the vocabulary.')

    parser.add_argument('--leave-unmasked-prob', '-u', type=float, default=0.1,
            help='Probability of a selected token being left'
            ' unmasked and unchanged.')

    parser.add_argument('--batch', '-b', type=int, default=8,
            help='Batch size of inputs to reconstruction model.')

    parser.add_argument('--num-samples', '-n', type=int, default=4,
            help='Number of augmented samples to generate for each'
            ' input example.')

    parser.add_argument('--max-tries', '-t', type=int, default=10,
            help='Number of tries to generate a unique sample'
            ' before giving up.')

    parser.add_argument('--min-len', type=int, default=4,
            help='Minimum length input for augmentation.')

    parser.add_argument('--max-len', type=int, default=512,
            help='Maximum length input for augmentation.')

    parser.add_argument('--topk', '-k', type=int, default=-1,
            help='Top k to use for sampling. -1 indicates'
            ' unrestricted sampling.')

    args = parser.parse_args()

    if args.shard >= args.num_shards:
        raise Exception('Shard number {} is too large for the number'
            ' of shards {}'.format(args.shard, args.num_shards))

    if not args.tokenizer:
        args.tokenizer = args.model


    gen_neighborhood(args)
