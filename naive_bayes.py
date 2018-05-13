# Naive Bayes 
"""
    P(A|B) = P(B|A) * P(A) / P(B)
    A = Class (SPAM or HAM)
    B = Word (1)

    -> mutiple words handled differently
"""

# P(A) -> probability of a class in a set of labeled emails
def class_prob(cl, _set):
    class_instances = 0
    
    # count how many times the class appears
    # in the set
    for _, c in _set:
        if c == cl:
            class_instances += 1
    
    # number of total instances
    total = len(_set)

    return class_instances / total

# P(B) -> probability of a word in a set of labeled emails
def word_prob(wd, _set):
    # the probability of a word is the sum of 
    # the probability of a word given a class times the probability of a class
    # for all clases (1, 0)

    # P(B) -> P(B|1) * P(1) + P(B|0) * P(0) 
    p_b = word_class_prob(wd, 1, _set) * class_prob(1, _set)
    p_b += word_class_prob(wd, 0, _set) * class_prob(0, _set)

    return p_b

# P(B|A) -> probability of a word given a class, inside a set
# of labeled emails
def word_class_prob(wd, cl, _set):
    word_instances = 0

    # gather all the words for the given class
    total_class_text = ""

    for w, c in _set:
        if c == cl:
            total_class_text += w + " "

    # convert text to list of words
    word_list = list(total_class_text.split())
    word_list_length = len(word_list)

    # count word appearances in word list
    for w in word_list:
        word = w.strip().upper()
        if word == wd:
            word_instances += 1

    return word_instances / word_list_length

    # P(A|B) -> Naive Bayes for a single word. Probabilty of a class
    # given a word, in a ste of labeled emails
def n_b(cl, wd, _set):
    word = wd.strip().upper()
    p_ba = word_class_prob(word, cl, _set)
    p_a = class_prob(cl, _set)
    p_b = word_prob(word, _set)

    return p_ba * p_a / p_b