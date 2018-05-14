# Naive Bayes 

# P(A) -> probability of a class in a set of labeled emails
def class_prob(cl, _set, smoothing):
    class_instances = 0
    
    # count how many times the class appears
    # in the set
    for _, c in _set:
        if c == cl:
            class_instances += 1
    
    # number of total instances
    total = len(_set)

    # smoothing: K = 1
    # cardinality: the number of different classes (2)
    # times K
    k = 0
    card = 0

    if smoothing:
        k = 1
        card = 2

    return (class_instances + k) / (total + card)

# P(B) -> probability of a word in a set of labeled emails
def word_prob(wd, _set, smoothing):
    # the probability of a word is the sum of 
    # the probability of a word given a class times the probability of a class
    # for all clases (1, 0)

    # P(B) -> P(B|1) * P(1) + P(B|0) * P(0) 
    p_b = word_class_prob(wd, 1, _set, smoothing) * class_prob(1, _set, smoothing)
    p_b += word_class_prob(wd, 0, _set, smoothing) * class_prob(0, _set, smoothing)

    return p_b

# P(B|A) -> probability of a word given a class, inside a set
# of labeled emails
def word_class_prob(wd, cl, _set, smoothing):
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

    # smoothing: K = 1
    # cardinality: the number of different words
    # times K
    k = 0
    card = 0

    if smoothing:
        total_text = ""
        for w, _ in _set:
            total_text += w + " "

        total_word_list = list(total_text.split())

        k = 1
        card = len(set(total_word_list))

    return (word_instances + k) / (word_list_length + card)

# P(A|B) -> Naive Bayes for a single word. Probabilty of a class
# given a word, in a set of labeled emails
def n_b_single_word(cl, wd, _set, smoothing):
    word = wd.strip().upper()
    p_ba = word_class_prob(word, cl, _set, smoothing)
    p_a = class_prob(cl, _set, smoothing)
    p_b = word_prob(word, _set, smoothing)

    try:
        result = p_ba * p_a / p_b
    except:
        print("ERROR: dividing by 0")
        result =  -1
    

    return result


# Complete Naive Bayes
def n_b(cl, wd, _set, smoothing=False):
    
    word_list = list(wd.split())
    word_list_length = len(word_list)

    # only one word probability
    if word_list_length == 1:
        return n_b_single_word(cl, wd, _set, smoothing)
    
    # for multiple words
    # P(A1) * P(B1|A1) * ... * P(Bn|A1) -> top: probability of the class times
    # the probability of all the words given that same class
    # / P(A1) * P(B1|A1) * ... * P(Bn|A1) + P(A2) * P(B1|A2) * ... * P(Bn|A2)
    # -> bottom: probability of a class times the probability off all words given that
    # same class, for all clases

    top = class_prob(cl, _set, smoothing)

    for w in word_list:
        word = w.strip().upper()
        top *= word_class_prob(word, cl, _set, smoothing)
    
    bottom = top

    # there are only two classes: 1 & 0
    other_class = 0
    if cl == 0:
        other_class = 1

    step = class_prob(other_class, _set, smoothing)

    for w in word_list:
        word = w.strip().upper()
        step *= word_class_prob(word, other_class, _set, smoothing)
    
    bottom += step

    try:
        result = top / bottom
    except:
        print("ERROR: dividing by 0")
        result = -1

    return result

