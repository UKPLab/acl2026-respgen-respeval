# Initialize static strings for the prompt template
task_description_noAIx = (
    "You are a research assistant helping authors prepare an author response for a paper under peer review. \n"
    "You will receive: "
    " - The reviewer's comment. "
    "Your task is to write a specific and convincing response addressing the reviewer's comment."    
)
style_prompt_noAIx_ARR = 'This author response is prepared during the rebuttal phase, before submitting any revisions (like in ARR process). Address the review comment specifically and convincingly.'
style_prompt_noAIx_ARR_PH = '''This author response is prepared during the rebuttal phase, before submitting any revisions (like in ARR process). Address the review comment specifically and convincingly. Use placeholders like '[author info: <description>]' if you need extra information from the author to address the review comment.'''
task_description_noAIx_item = (
    "You are a research assistant helping authors prepare an author response for a paper under peer review. \n"
    "You will receive: "
    " - The reviewer's comment. And extracted items from the review comment, including questions, criticisms and requests."
    "Your task is to write a specific and convincing response addressing the reviewer's comment and the items. Make the response coherent, fluent and human-like, without necessarily listing the items."    
)

task_description_wAIx_ARR = (
    "You are a research assistant helping authors prepare an author response for a paper under peer review. \n"
    "You will receive: "
    " - The reviewer's comment. "
    " - The author's additional input regarding the comment. \n"
    "Your task is to write a clear and convincing response addressing the reviewer's comment and the items."    
)
style_prompt_wAIx_ARR = 'This author response is prepared during the rebuttal phase, before submitting any revisions (like in ARR process). Address the review comment specifically and convincingly. You should use the additional author input to address the review comment if they are useful, and may outline future planned changes in the final version if relevant but do not refer to completed revisions.'

style_prompt_wAIx_ARR_PH = '''This author response is prepared during the rebuttal phase, before submitting any revisions (like in ARR process). Address the review comment specifically and convincingly. You should use the additional author input to address the review comment if they are useful, and may outline future planned changes in the final version if relevant but do not refer to completed revisions. Use placeholders like '[author info: <description>]' if you need extra information from the author to address the review comment.''' \

task_description_wAIx_ARR_item = (
    "You are a research assistant helping authors prepare an author response for a paper under peer review. \n"
    "You will receive: "
    " - The reviewer's comment. And extracted items from the review comment, including questions, criticisms and requests."
    " - The author's additional input regarding the comment. \n"
    "Your task is to write a clear and convincing response addressing the reviewer's comment and the items. Make the response coherent, fluent and human-like, without necessarily listing the items."    
)

task_description_wAIx_ARR_item_selfplan = (
    '''For each given item, first create a response action plan with a sequence of labels in \
    ['answer question','refute question','mitigate importance of the question',\
                        'concede criticism','accept for future work','reject criticism','mitigate criticism','contradict assertion',\
                        'reject request','task has been done','task will be done in next version','accept for future work', \
                        'social', 'structure','summarize','follow-up question', 'other']. \
    Then write a response addressing the review comment and the items based on the action plan. \
    Always start the plan with "###Plan: \n" and list the labels for each item, like this:
    ###Plan: \n
    --- questions: #1: answer question, accept for future work; \n
    --- criticisms: none; \n
    --- requests: #1: task will be done in next version, mitigate criticism; #2: contradict assertion, mitigate criticism;  \n
    then start the response with "###Response: \n". \
    '''   
)
task_description_wAIx_ARR_item_authorplan = (
    '''Write a response addressing the review comment and the items based on the given response action plan.
    '''   
)


# structure components for author input
edit_start = "- Edit :"
edit_end = ""
edit_label_start = " -- The edit's action and intent are:"
edit_label_end = ""
edit_sec_start = " -- The edit occurs in the old/new sections:"
edit_sec_end = ""
edit_old_start = " -- The old text of the edit is:"
edit_old_end = ""
edit_new_start = " -- The new text of the edit is:"
edit_new_end = ""
review_start = "- The review comment is:"
review_end = ""
label_start = "- The true label is:"
label_end = ""

edit_start_st = "<edit>"
edit_end_st = "</edit>"
edit_label_start_st = " <edit_label>"
edit_label_end_st = "</edit_label>"
edit_sec_start_st = " <edit_section>"
edit_sec_end_st = "</edit_section>"
edit_old_start_st = " <edit_old>"
edit_old_end_st = "</edit_old>"
edit_new_start_st = " <edit_new>"
edit_new_end_st = "</edit_new>"
review_start_st = "<review>"
review_end_st = "</review>"
label_start_st = "<label>"
label_end_st = "</label>"

PROMPT_ST_DIC = {'nl': {'edit_start': edit_start, 'edit_end': edit_end, 'edit_label_start': edit_label_start, 'edit_label_end': edit_label_end, 'edit_sec_start': edit_sec_start, 'edit_sec_end': edit_sec_end, 'edit_old_start': edit_old_start, 'edit_old_end': edit_old_end, 'edit_new_start': edit_new_start, 'edit_new_end': edit_new_end, 'review_start': review_start, 'review_end': review_end, 'label_start': label_start, 'label_end': label_end},
                 'st': {'edit_start': edit_start_st, 'edit_end': edit_end_st, 'edit_label_start': edit_label_start_st, 'edit_label_end': edit_label_end_st, 'edit_sec_start': edit_sec_start_st, 'edit_sec_end': edit_sec_end_st, 'edit_old_start': edit_old_start_st, 'edit_old_end': edit_old_end_st, 'edit_new_start': edit_new_start_st, 'edit_new_end': edit_new_end_st, 'review_start': review_start_st, 'review_end': review_end_st, 'label_start': label_start_st, 'label_end': label_end_st}}