response_label_classes = {
    "Cooperative": [
        "answer question",
        "task has been done",
        "task will be done in next version",
        "accept for future work",
        "concede criticism",
    ],
    "Defensive": [
        "refute question",
        "reject criticism",
        "contradict assertion",
        "reject request",
    ],
    "Hedge": [
        "mitigate importance of the question",
        "mitigate criticism",
    ],
    "Social": [
        "social",
    ],
    "NonArg": [
        "follow-up question",
        "structure",
        "summarize",
        "other",
    ],
}


system_prompt_dict = {
    "item-link-label-score": '''Input: a peer review comment and an author response. \n
                        Tasks: Extract questions (Q), weakness criticisms (C), and requests (R) from the review. \
                        For each Q/C/R item, find all related response sentences targeting it (can be none).\n
                        Response Labels ∈ ['answer question','refute question','mitigate importance of the question',\n
                        'concede criticism','reject criticism','mitigate criticism','contradict assertion',\n
                        'reject request','task has been done','task will be done in next version','accept for future work', \n
                        'social', 'structure','summarize','follow-up question', 'other'] \n
                        Scoring: For each Q/C/R item, set response_conv_score ∈ [0,1] (how convincing the response is to that item) and set  response_spec_score ∈ [0,1] (how specific the response is to that item). If the item’s response list is empty, use 0. Round to 2 decimals.\n\n
                        Finally, list the rest response sentences that do not target any Q/C/R item and label them (consider the last 5 non-argumentative labels). \n
                        Important: Output JSON only, no prose, no reasoining. Keep field names exact. Empty arrays are allowed. Preserve the review’s original order of first appearance. \n
                        {
                        "questions": [ 
                            {
                            "review_text": [<list of review sentences about the same point>],
                            "response": [
                                {"text": <response text>, "labels": [<Q labels>]}
                            ],
                            "response_conv_score": <float 0-1>,
                            "response_spec_score": <float 0-1>
                            }
                        ],
                        "criticisms": [
                            {
                            "review_text": [<list of review sentences about the same point>],
                            "response": [
                                {"text": <response text>, "labels": [<C labels>]}
                            ],
                            "response_conv_score": <float 0-1>,
                            "response_spec_score": <float 0-1>
                            }
                        ],
                        "requests": [
                            {
                            "review_text": [<list of review sentences about the same point>],
                            "response": [
                                {"text": <response text>, "labels": [<R labels>]}
                            ],
                            "response_conv_score": <float 0-1>,
                            "response_spec_score": <float 0-1>
                            }
                        ],
                        "other_responses": [
                            {"text": <response text>, 
                            "labels": [<labels>]}
                        ]
                        }''',
     "link-label-score": '''Input: a peer review comment and an author response. And extracted questions (Q), weakness criticisms (C), and requests (R) from the review, provided in json.\n
                        Tasks: Find all related response sentences targeting each Q/C/R  item (can be none).\n
                        Response Labels ∈ ['answer question','refute question','mitigate importance of the question',\n
                        'concede criticism','accept for future work','reject criticism','mitigate criticism','contradict assertion',\n
                        'reject request','task has been done','task will be done in next version','accept for future work', \n
                        'social', 'structure','summarize','follow-up question', 'other'] \n
                        Scoring: For each Q/C/R item, set response_conv_score ∈ [0,1] (how convincing the response is to that item) and set  response_spec_score ∈ [0,1] (how specific the response is to that item). If the item’s response list is empty, use 0. Round to 2 decimals.\n\n
                        Finally, list the rest response sentences that do not target any Q/C/R item and label them (consider the last 5 non-argumentative labels). \n
                        Important: Use the given json with identified items, update the 'response' and scoring fields only without changing other fields and structure. Output JSON only, no prose, no reasoning. Keep field names exact. Empty arrays are allowed. Preserve the review’s and response’s original order of first appearance. \n
                        {
                        "questions": [ 
                            {
                            "review_text": [<list of review sentences about the same point>],
                            "response": [
                                {"text": <response text>, "labels": [<Q labels>]}
                            ],
                            "response_conv_score": <float 0-1>,
                            "response_spec_score": <float 0-1>
                            }
                        ],
                        "criticisms": [
                            {
                            "review_text": [<list of review sentences about the same point>],
                            "response": [
                                {"text": <response text>, "labels": [<C labels>]}
                            ],
                            "response_conv_score": <float 0-1>,
                            "response_spec_score": <float 0-1>
                            }
                        ],
                        "requests": [
                            {
                            "review_text": [<list of review sentences about the same point>],
                            "response": [
                                {"text": <response text>, "labels": [<R labels>]}
                            ],
                            "response_conv_score": <float 0-1>,
                            "response_spec_score": <float 0-1>
                            }
                        ],
                        "other_responses": [
                            {"text": <response text>, 
                            "labels": [<other labels>]}
                        ]
                        }''',

}