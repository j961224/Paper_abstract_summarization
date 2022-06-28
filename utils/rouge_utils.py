import six
import numpy as np
import collections
from rouge_score import scoring

def create_ngrams(tokens, n):
    """Creates ngrams from the given list of tokens.
	Args:
		tokens: A list of tokens from which ngrams are created.
		n: Number of tokens to use, e.g. 2 for bigrams.
	Returns:
		A dictionary mapping each bigram to the number of occurrences.
	"""
    ngrams = collections.Counter()
    for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
        ngrams[ngram] += 1
    return ngrams


def score_lcs(reference_tokens, prediction_tokens):
	"""Computes LCS (Longest Common Subsequence) rouge scores.
	Args:
		reference_tokens: Tokens from the reference text.
		prediction_tokens: Tokens from the predicted text.
	Returns:
		A Score object containing computed scores.
	"""

	if not reference_tokens or not prediction_tokens:
		return scoring.Score(precision=0, recall=0, fmeasure=0)

	# Compute length of LCS from the bottom up in a table (DP appproach).
	lcs_tables = lcs_table(reference_tokens, prediction_tokens)
	lcs_length = lcs_tables[-1][-1] ## Longest Common Subsequence 수 

	precision = lcs_length / len(prediction_tokens)
	recall = lcs_length / len(reference_tokens)
	fmeasure = scoring.fmeasure(precision, recall)

	return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)

def lcs_table(reference_tokens, prediction_tokens):
	"""Create 2-d LCS score table."""
	rows = len(reference_tokens)
	cols = len(prediction_tokens)
	lcs_tables = [[0] * (cols + 1) for _ in range(rows + 1)]
	for i in range(1, rows + 1):
		for j in range(1, cols + 1):
			if reference_tokens[i - 1] == prediction_tokens[j - 1]:
				lcs_tables[i][j] = lcs_tables[i - 1][j - 1] + 1
			else:
				lcs_tables[i][j] = max(lcs_tables[i - 1][j], lcs_tables[i][j - 1])
	return lcs_tables


def backtrack_norec(t, ref, pred):
	"""Read out LCS."""
	i = len(ref)
	j = len(pred)
	lcs = []
	while i > 0 and j > 0:
		if ref[i - 1] == pred[j - 1]:
			lcs.insert(0, i-1)
			i -= 1
			j -= 1
		elif t[i][j - 1] > t[i - 1][j]:
			j -= 1
		else:
			i -= 1
	return lcs


def summary_level_lcs(reference_tokens_list, prediction_tokens_list):
	"""ROUGE: Summary-level LCS, section 3.2 in ROUGE paper.
	Args:
	reference_tokens_list: list of tokenized reference sentences
	prediction_tokens_list: list of tokenized prediction sentences
	Returns:
	summary level ROUGE score
	"""
	if not reference_tokens_list or not prediction_tokens_list:
		return scoring.Score(precision=0, recall=0, fmeasure=0)

	m = sum(map(len, reference_tokens_list))
	n = sum(map(len, prediction_tokens_list))
	if not n or not m:
		return scoring.Score(precision=0, recall=0, fmeasure=0)

	# get token counts to prevent double counting
	token_cnts_r = collections.Counter()
	token_cnts_c = collections.Counter()
	for s in reference_tokens_list:
		# s is a list of tokens
		token_cnts_r.update(s)
	for s in prediction_tokens_list:
		token_cnts_c.update(s)

	hits = 0
	for ref in reference_tokens_list:
		lcs = union_lcs(ref, prediction_tokens_list) ## ref가 여러 문장의 prediction 문장의 for문을 돌면서 일치하는 것 찾아줌
		# Prevent double-counting:
		# The paper describes just computing hits += len(_union_lcs()),
		# but the implementation prevents double counting. We also
		# implement this as in version 1.5.5.
		for t in lcs:
			if token_cnts_c[t] > 0 and token_cnts_r[t] > 0:
				hits += 1
				token_cnts_c[t] -= 1
				token_cnts_r[t] -= 1

	recall = hits / m
	precision = hits / n
	fmeasure = scoring.fmeasure(precision, recall)
	return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)


def union_lcs(ref, prediction_tokens_list):
	"""Find union LCS between a ref sentence and list of predicted sentences.
	Args:
	ref: list of tokens
	prediction_tokens_list: list of list of indices for LCS into reference summary
	Returns:
	List of tokens in ref representing union LCS.
	"""
	lcs_list = [lcs_ind(ref, pred) for pred in prediction_tokens_list]
	return [ref[i] for i in find_union(lcs_list)]


def find_union(lcs_list):
	"""Finds union LCS given a list of LCS."""
	return sorted(list(set().union(*lcs_list)))


def lcs_ind(ref, pred):
	"""Returns one of the longest lcs."""
	t = lcs_table(ref, pred)
	return backtrack_norec(t, ref, pred)


def score_ngrams(reference_ngrams, prediction_ngrams):
	"""Compute n-gram based rouge scores.
	Args:
	reference_ngrams: A Counter object mapping each ngram to number of
		occurrences for the reference text.
	prediction_ngrams: A Counter object mapping each ngram to number of
		occurrences for the prediction text.
	Returns:
	A Score object containing computed scores.
	"""

	intersection_ngrams_count = 0
	for ngram in six.iterkeys(reference_ngrams): ## python 3 ".keys()"
		intersection_ngrams_count += min(reference_ngrams[ngram], prediction_ngrams[ngram]) ## 겹치는 n_gram  counting (True positive)
	reference_ngrams_count = sum(reference_ngrams.values())
	prediction_ngrams_count = sum(prediction_ngrams.values())

	precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
	recall = intersection_ngrams_count / max(reference_ngrams_count, 1)
	fmeasure = scoring.fmeasure(precision, recall) ## f1 score 계산

	return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure) ## Output formatting