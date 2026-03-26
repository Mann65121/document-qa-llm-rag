def score_text(text, query, doc_frequencies, total_docs):
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    text_tokens = tokenize(text)
    if not text_tokens:
        return 0.0

    counts = Counter(text_tokens)
    token_set = set(text_tokens)
    overlap = 0.0

    for token in query_tokens:
        if token not in token_set:
            continue
        inverse_frequency = math.log((total_docs + 1) / (1 + doc_frequencies.get(token, 0))) + 1
        overlap += counts[token] * inverse_frequency

    phrase_bonus = text.lower().count(query.lower()) * 3.5

    question_word = query_type(query)
    hint_bonus = 0.0
    if question_word in QUESTION_HINTS:
        hint_bonus = sum(0.4 for hint in QUESTION_HINTS[question_word] if hint in text.lower())

    density = overlap / math.sqrt(len(text_tokens))
    length_penalty = 1 / (1 + len(text_tokens) * 0.01)

    return (overlap + density + phrase_bonus + hint_bonus) * length_penalty