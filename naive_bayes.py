# Naive Bayes 
"""
    P(A|B) = P(B|A) * P(A) / P(B)
    A = Class (SPAM or HAM)
    B = Word (1)

    -> mutiple words handled differently
"""

# P(A) -> probability of a class in a set of labeled emails
def class_prob(cl, set):
    class_instances = 0
    
    # count how many times the class appears
    # in the set
    for _, c in set:
        if c == cl:
            class_instances += 1
    
    # number of total instances
    total = len(set)

    return class_instances / total

# P(B) -> probability of a word in a set of labeled emails
def word_prob(wd, set):
    word_instances = 0

    # all emails contain a string of text
    # with multiple words separated by spaces
    # add them all together
    total_text = ""

    for w, _ in set:
        total_text += w + " "
    
    # convert text to list of words
    word_list = list(total_text.split())
    word_list_length = len(word_list)

    # count word appearances in word list
    for w in word_list:
        word = w.strip().upper()
        if word == wd:
            word_instances += 1
    
    return word_instances / word_list_length

# P(B|A) -> probability of a word given a class, inside a set
# of labeled emails
def word_class_prob(wd, cl, set):
    word_instances = 0

    # gather all the words for the given class
    total_class_text = ""

    for w, c in set:
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
def n_b(cl, wd, set):
    word = wd.strip().upper()
    return word_class_prob(word, cl, set) * class_prob(cl, set) / word_prob(word, set)