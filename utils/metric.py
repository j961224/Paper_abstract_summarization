import re
import six
import numpy as np
from tqdm import tqdm
from rouge_score import scoring, rouge_scorer
from .rouge_utils import (
    create_ngrams, 
    score_lcs, 
    score_ngrams, 
    summary_level_lcs
)

def compute(predictions, references, tokenizer, use_agregator=True):
    # if rouge_types is None:
    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        
    scorer = KoreanRouge(rouge_types=rouge_types, tokenizer=tokenizer)
    if use_agregator:
        aggregator = scoring.BootstrapAggregator() #################
    else:
        scores = []
        
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        if use_agregator:
            aggregator.add_scores(score)
        else:
            scores.append(score)
    
    if use_agregator:
        result = aggregator.aggregate()
    else:
        result = {}
        for key in scores[0]:
            result[key] = list(score[key] for score in scores)
    return result

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	
    # Some simple post-processing
    result = compute(predictions=decoded_preds, references=decoded_labels, tokenizer=tokenizer)
	
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

class KoreanRouge(rouge_scorer.RougeScorer) :
    # """ 
	# https://github.com/google-research/google-research/blob/master/rouge
	# google-research의 RougeScorer를 상속.
	# 한국어 tokenizer를 적용한 rouge score를 계산하도록 custom.
	# Args:
	# 	rouge_type (list[str]): 원하는 Rouge Score type list
	# 	tokenizer (PreTrainedTokenizerBase): 모델에 맞는 tokenizer
	# """
    def __init__(self, rouge_types, tokenizer) :
        super(KoreanRouge).__init__()
        self.rouge_types = rouge_types
        self.tokenizer = tokenizer
    
    def score(self, references, prediction):
        # """Calculates rouge scores between the references and prediction.
		# Args:
		# 	references: Text containing the references (ground truth) text.
		# 	prediction: Text containing the predicted text.
		# Returns:
		# 	A dict mapping each rouge type to a Score object.
		# Raises:
		# 	ValueError: If an invalid rouge type is encountered.
		# """

		# Pre-compute references tokens and prediction tokens for use by different
		# types, except if only "rougeLsum" is requested.
        if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
            reference_tokens = None
            prediction_tokens = None
        else:
            reference_tokens = self.tokenizer.tokenize(references)
            prediction_tokens = self.tokenizer.tokenize(prediction)
        result = {}
        
        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                # Rouge from longest common subsequences.
                scores = score_lcs(reference_tokens, prediction_tokens)
            elif rouge_type == "rougeLsum":
                # Note: Does not support multi-line text.
                def get_sents(text):
                    # Assume sentences are separated by newline.
                    sents = six.ensure_str(text).split("\n")
                    sents = [x for x in sents if len(x)]
                    return sents
                
                ## \n로 구분된 문장을 split하여 lcs
                reference_tokens_list = [
					self.tokenizer.tokenize(s) for s in get_sents(references)] ## muti-line split 후 tokenizer
                prediction_tokens_list = [
					self.tokenizer.tokenize(s) for s in get_sents(prediction)] ## muti-line split 후 tokenizer
                scores = summary_level_lcs(reference_tokens_list, prediction_tokens_list)
            
            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)): ## six.ensure_str: rouge type을 강제적으로 str type 변환
                # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)
                reference_ngrams = create_ngrams(reference_tokens, n)
                prediction_ngrams = create_ngrams(prediction_tokens, n)
                scores = score_ngrams(reference_ngrams, prediction_ngrams)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)
            result[rouge_type] = scores
        
        return result
