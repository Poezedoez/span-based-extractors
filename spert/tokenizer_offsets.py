from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import regex as re


# even when tokens themselves have whitespace, for most tasks we want our offsets to not include the whitespace
def whitespace_reduce(offsets, text):
    for ti in range(offsets.shape[0]):
        nstart = offsets[ti, 0]
        nend = offsets[ti, 1]
        while nstart < nend and text[nstart].isspace():
            nstart += 1
        while nstart < nend and text[nend - 1].isspace():
            nend -= 1
        if nstart < nend:
            # assert text[offsets[ti, 0]:offsets[ti, 1]].strip() == text[nstart:nend]
            offsets[ti, 0] = nstart
            offsets[ti, 1] = nend


UNMATCHABLE_TOKENS = [u'�']  # consider putting in tokenization_utils, so subclasses can override


def match_back_by_length(tlens, offsets, tstart, tend):
    # split the chunk, dividing the chunk's span proportionally among the tokens
    if tend - tstart <= 1:
        return
    # when this is called, all offsets[tstart:tend] are the same (the same chunk)
    text_end = offsets[tstart, 1]
    prev_end = offsets[tstart, 0]
    tok_len_remaining = sum(tlens[tstart:tend]) + 0.1
    for ci in range(tstart, tend):
        text_remaining = text_end - prev_end
        token_len = tlens[ci] + 0.1 / (tend - tstart)
        offsets[ci, 0] = prev_end
        if ci < tend - 1:
            scale = text_remaining / tok_len_remaining
            leni = int(round(scale * token_len))
            offsets[ci, 1] = min(text_end, offsets[ci, 0] + leni)
            prev_end = offsets[ci, 1]
        tok_len_remaining -= token_len


def match_back_by_text(tokens, text, tlens, offsets, tstart, tend):
    match_start = tstart
    match_end = tend - 1
    txt_start = offsets[match_start, 0]
    txt_end = offsets[match_end, 1]
    orig_txt_start = txt_start
    orig_txt_end = txt_end
    # try to find the token strings in the original text
    text = text.lower()  # any other length preserving normalizing?
    while match_start <= match_end and txt_start < txt_end:
        findndx = text.find(tokens[match_start].lower(), txt_start, txt_end)
        pre_skip = findndx - txt_start
        rfindndx = text.rfind(tokens[match_end].lower(), txt_start, txt_end)
        post_skip = txt_end - rfindndx - len(tokens[match_end])
        # do we skip more of the text by matching the first token or the last token?
        # we want to greedily make good matches
        if findndx != -1 and (rfindndx == -1 or pre_skip <= post_skip):
            offsets[match_start, 0] = findndx
            offsets[match_start, 1] = offsets[match_start, 0] + len(tokens[match_start])
            txt_start = offsets[match_start, 1]
            match_start += 1
        elif rfindndx != -1:
            offsets[match_end, 0] = rfindndx
            offsets[match_end, 1] = offsets[match_end, 0] + len(tokens[match_end])
            txt_end = offsets[match_end, 0]
            match_end -= 1
        else:
            break
    # we matched everything. good job!
    if match_start > match_end:
        return
    # we messed up, hand it all to the match_by_length
    if txt_start > txt_end or sum(tlens[match_start:match_end + 1]) > txt_end - txt_start:
        txt_start = orig_txt_start
        txt_end = orig_txt_end
        match_start = tstart
        match_end = tend - 1
    # anything leftover we match by length
    for leftover in range(match_start, match_end + 1):
        offsets[leftover, 0] = txt_start
        offsets[leftover, 1] = txt_end
    match_back_by_length(tlens, offsets, match_start, match_end + 1)
    # DEBUG: show what we came up with
    # print('*' * 10)
    # print(f'{text[offsets[tstart, 0]:offsets[tend-1, 1]]}')
    # for ti in range(tstart, tend):
    #    print(f'"{tokens[ti]}" "{text[offsets[ti, 0]:offsets[ti, 1]]}"')
    # print('*' * 10)


# when a chunk becomes multiple tokens, we find what spans of the chunk each token corresponds to
def multitoken_chunk_offsets(tokens, text, tlens, offsets, tstart, tend):
    if tend - tstart <= 1:
        return
    if offsets[tstart, 1] - offsets[tstart, 0] == sum(tlens[tstart:tend]):
        # the sum of token length is the chunk length, just chop it up
        match_back_by_length(tlens, offsets, tstart, tend)
    else:
        # try matching the token text from the start and end, then punt on the middle
        match_back_by_text(tokens, text, tlens, offsets, tstart, tend)


# handle chunks that have multiple tokens
def tokens_in_chunks(tokens, text, offsets):
    assert len(tokens) == offsets.shape[0]
    same_chunk_start = 0
    tlens = [len(t) if t not in UNMATCHABLE_TOKENS else 0 for t in tokens]
    for i in range(1, len(tokens)):
        if not np.array_equal(offsets[same_chunk_start, :], offsets[i, :]):
            multitoken_chunk_offsets(tokens, text, tlens, offsets, same_chunk_start, i)
            same_chunk_start = i
    multitoken_chunk_offsets(tokens, text, tlens, offsets, same_chunk_start, len(tokens))


def finalize_token_offsets(detokenized_tokens, text, offsets):
    """
    detokenized_tokens are tokens that can usually be matched by to the text (best effort).
    The text is just the original text, before any cleaning by the tokenizer.
    The offsets are currently chunk offsets, multiple tokens have the same offsets.
    We will give the tokens non-overlapping offsets that match the text as closely as possible.
    :param detokenized_tokens: tokens from the tokenizer, with artifacts removed to match the original text better
    :param text: the original text, before our preprocessing got its hands on it
    :param offsets: len(tokens) x 2, giving start and end for each token,
                    initially this is the start and end of the chunk
    :return: the numpy array of offsets is modified
    """
    # adjust token offsets to not include leading/trailing whitespace
    whitespace_reduce(offsets, text)
    # find offsets for sub-chunk tokens
    tokens_in_chunks(detokenized_tokens, text, offsets)

    # asserts on non-overlapping spans with start <= end
    for i in range(offsets.shape[0]):
        assert offsets[i, 0] <= offsets[i, 1]
        if i > 0:
            assert offsets[i - 1, 1] <= offsets[i, 0]


def detokenize_for_offsets(self, tok):
    """
    Remove any tokenization artifacts for sub-word tokens.
    Used by tokenize_with_offsets to match tokens back in the original text.
     (like ## prefix for BERT)
    :param tok: the token from the tokenizer
    :return: the token as it would have looked (best approximation) in the original text
    """
    return tok.strip()


def tokenize_with_offsets(self, text, initial_space=False, **kwargs):
    """
    The utils_squad approach to getting from tokens to original text.
    We chop up into 'chunks' and pass each of these to the tokenizer.
    We know our offsets will never be terrible, since we know the chunk the token comes from.
    If multiple tokens come back for a chunk, we divide the span up proportional to token length.
    Known issues:
      in OpenAIGPTTokenizer, XLMTokenizer we lose '\n</w>'
         compare 'hello \n there'
         possible solution: remove the .strip() in the preprocessing OR surround input with special tokens
      in TransfoXLTokenizer
         test add_eos=False, add_double_eos=False
    :param text: the text to tokenize
    :param initial_space: should we include an initial_space when tokenizing the text?
    :param kwargs: passed to the underlying tokenization implementation
    :return: tokens, offsets; offsets is len(tokens) x 2 numpy array of character offsets [begin,one-past-end)
    """
    tok_to_chunk_index = []
    tokens = []

    # add the tokenized chunk to the tokens along with the chunk index they correspond to
    def tokenize_with_offsets_chunk(chunks_with_start, chunk_start, **kwargs):
        chunks = [t[0] for t in chunks_with_start]
        for i, chunk in enumerate(chunks):
            sub_tokens = self._tokenize(chunk, **kwargs)
            for sub_token in sub_tokens:
                tok_to_chunk_index.append(i + chunk_start)
                tokens.append(sub_token)

    def chunkify(offset, text, exclude_empty=False, strip=True):
        # exclude_empty mirrors the tokenize method's policy of discarding leading/trailing whitespace segments
        # when they come before/after a special token
        space = 0
        # maybe strip spaces
        if strip:
            while space < len(text) and text[space].isspace():
                space += 1
            text = text.strip()
            offset += space
        if exclude_empty and not text:
            return []
        # maybe add an initial space, like for GPT2
        if initial_space:
            text = ' ' + text
            offset -= 1
        chunks = [(m.group(0), offset + m.start(), offset + m.end()) for m in re.finditer(self.splitter_pat, text)]
        # our imaginary space may not be in the original text though
        if initial_space and space == 0 and chunks[0][1] == offset:
            chunks[0] = chunks[0][0], chunks[0][1] + 1, chunks[0][2]
        return chunks

    # handle self.added_tokens_encoder and self.all_special_tokens
    added_tokens = list(self.added_tokens_encoder.keys()) + self.all_special_tokens
    added_tokens.sort()  # sort to canonicalize the resulting special_pat_str
    special_pat_str = '|'.join([re.escape(st) for st in added_tokens])
    if special_pat_str:
        # create or recreate the special token regex
        if special_pat_str != self.special_pat_str:
            self.special_pat = re.compile(special_pat_str)
            self.special_pat_str = special_pat_str
        chunks_offsets = []
        prev_end = 0
        for sm in re.finditer(self.special_pat, text):
            # add the tokens before the special token
            sub_chunks = chunkify(prev_end, text[prev_end:sm.start()], exclude_empty=not chunks_offsets)
            tokenize_with_offsets_chunk(sub_chunks, len(chunks_offsets), **kwargs)
            chunks_offsets.extend(sub_chunks)
            # add the special token
            tokens.append(sm.group(0))
            tok_to_chunk_index.append(len(chunks_offsets))
            chunks_offsets.append((sm.group(0), sm.start(), sm.end()))
            prev_end = sm.end()
        # add remaining tokens
        sub_chunks = chunkify(prev_end, text[prev_end:], exclude_empty=True)
        tokenize_with_offsets_chunk(sub_chunks, len(chunks_offsets), **kwargs)
        chunks_offsets.extend(sub_chunks)
    else:
        chunks_offsets = chunkify(0, text, strip=False)  # TODO: behavior copied from tokenize, correct?
        tokenize_with_offsets_chunk(chunks_offsets, 0, **kwargs)
    # convert the offsets to np array
    offsets = np.zeros((len(tok_to_chunk_index), 2), dtype=np.int32)
    for ti, ci in enumerate(tok_to_chunk_index):
        offsets[ti, 0] = chunks_offsets[ci][1]
        offsets[ti, 1] = chunks_offsets[ci][2]
    # when chunks have multiple tokens, we need to find the proper span for each
    finalize_token_offsets([self.detokenize_for_offsets(t) for t in tokens], text, offsets)
    return tokens, offsets
