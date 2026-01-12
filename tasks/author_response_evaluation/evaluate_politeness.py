from tqdm import tqdm
import json
from transformers import pipeline
import numpy as np
import nltk
nltk.download('punkt_tab')

def evaluate_politeness(df, eval_gold=True):
    """ Evaluate the politeness of the author response generation task.
    Args:
        df: DataFrame with columns 'true' and 'pred'
    Returns:
        A dictionary with politeness evaluation metrics.
    """
    eval_cols = ['polite_true', 'polite_sent_mean_true', 'polite_sent_prop_true', 
                 'polite_pred', 'polite_sent_mean_pred', 'polite_sent_prop_pred']
    # add new columns to df
    for col in eval_cols:
        if col not in df.columns:
            df[col] = None

    # result sample: {'label': 'polite', 'score': 0.9950300455093384}
    classifier = pipeline("text-classification", 
                          model="Genius1237/xlm-roberta-large-tydip",
                          tokenizer="Genius1237/xlm-roberta-large-tydip",
                          truncation=True,     
                          max_length=512 
                          )

    for i, row in tqdm(df.iterrows()):
        print(f"Evaluating politeness of row {i}")
        true = row['true'].strip()
        pred = row['pred'].strip()

        if eval_gold:
            # Politeness evaluation for true response as a whole
            res = classifier(true)
            df.at[i, 'polite_true'] = round(res[0]['score'], 4)

        # Politeness evaluation for predicted response as a whole
        res = classifier(pred)
        df.at[i, 'polite_pred'] = round(res[0]['score'], 4)


        if eval_gold:
            # Politeness evaluation for true response, split into sentences
            true_sentences = nltk.sent_tokenize(true)
            true_sent_props = [classifier(sent) for sent in true_sentences]
            polite_sents = [res[0]['label'] for res in true_sent_props if res[0]['label'] == 'polite']
            df.at[i, 'polite_sent_prop_true'] = round(len(polite_sents) / len(true_sentences), 4)
            df.at[i, 'polite_sent_mean_true'] = round(np.mean([res[0]['score'] for res in true_sent_props]), 4)
            true_sent_res = [{'sentence': sent, 'label': res[0]['label'], 'score': res[0]['score']} for sent, res in zip(true_sentences, true_sent_props)]
        else:
            true_sent_res = []
        # Politeness evaluation for predicted response, split into sentences
        pred_sentences = nltk.sent_tokenize(pred)
        for sent in pred_sentences:
            res = classifier(sent)
            if res[0]['label'] != 'polite':
                print(f"##### Politeness evaluation for predicted sentence: {sent} - \n {res}")
        pred_sent_props = [classifier(sent) for sent in pred_sentences]
        polite_sents = [res[0]['label'] for res in pred_sent_props if res[0]['label'] == 'polite']
        df.at[i, 'polite_sent_prop_pred'] = round(len(polite_sents) / len(pred_sentences), 4)
        df.at[i, 'polite_sent_mean_pred'] = round(np.mean([res[0]['score'] for res in pred_sent_props]), 4)
        pred_sent_res = [{'sentence': sent, 'label': res[0]['label'], 'score': res[0]['score']} for sent, res in zip(pred_sentences, pred_sent_props)]

    # Calculate overall scores, take 4 decimal places
    overall_scores_report = {
        'polite_true': round(df['polite_true'].mean(), 4), #results for true human responses
        'polite_sent_mean_true': round(df['polite_sent_mean_true'].mean(), 4),
        'polite_sent_prop_true': round(df['polite_sent_prop_true'].mean(), 4),
        'polite_pred': round(df['polite_pred'].mean(), 4), #results for model generated responses
        'polite_sent_mean_pred': round(df['polite_sent_mean_pred'].mean(), 4),
        'polite_sent_prop_pred': round(df['polite_sent_prop_pred'].mean(), 4),
    }
    return df, overall_scores_report, {
        'true_sent_res': true_sent_res,
        'pred_sent_res': pred_sent_res
    }