from rouge_score import rouge_scorer
from bert_score import score as bert_score

def evaluate_basics(df):
    """
    Evaluate the basic metrics for author response generation.
    Args:
        df: DataFrame with columns 'true' and 'pred'
    Returns:
        A dictionary with basic evaluation metrics.
    """
    eval_cols = ['basic_rouge1', 'basic_rouge2', 'basic_rougeL', 'basic_bert_P', 'basic_bert_R', 'basic_bert_F1']
    # add new columns to df
    for col in eval_cols:
        if col not in df.columns:
            df[col] = None

    # calculate basic scores for each row
    for i, row in df.iterrows():
        true = row['true'].strip()
        pred = row['pred'].strip()
        # Calculate ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(true, pred)
        for key, value in scores.items():
            df.at[i, f'basic_{key}'] = round(value.fmeasure, 4)

        # Calculate BERT scores
        P, R, F1 = bert_score([pred], [true], lang="en", verbose=False)
        df.at[i, 'basic_bert_P'] = round(P.item(), 4)
        df.at[i, 'basic_bert_R'] = round(R.item(), 4)
        df.at[i, 'basic_bert_F1'] = round(F1.item(), 4)

    # Calculate overall scores, take 4 decimal places
    overall_scores_report = {
        'basic_rouge1': round(df['basic_rouge1'].mean(), 4),
        'basic_rouge2': round(df['basic_rouge2'].mean(), 4),
        'basic_rougeL': round(df['basic_rougeL'].mean(), 4),
        'basic_bert_P': round(df['basic_bert_P'].mean(), 4),
        'basic_bert_R': round(df['basic_bert_R'].mean(), 4),
        'basic_bert_F1': round(df['basic_bert_F1'].mean(), 4),
    }
    return df, overall_scores_report
