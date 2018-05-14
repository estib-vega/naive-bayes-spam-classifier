from test_mails import prob71 as mails
from naive_bayes import n_b

if __name__ == "__main__":
    prob_1 = n_b(1, 'free', mails)
    prob_2 = n_b(0, 'free', mails)
    prob_3 = n_b(1, 'money', mails)
    prob_4 = n_b(0, 'money', mails)
    prob_5 = n_b(1, 'free money', mails)
    prob_6 = n_b(0, 'free money', mails)

    print("Probability of class 1 given 'free':", prob_1)
    print("Probability of class 0 given 'free':", prob_2)
    print("total:", prob_1 + prob_2)
    print("Probability of class 1 given 'money':", prob_3)
    print("Probability of class 0 given 'money':", prob_4)
    print("total:", prob_3 + prob_4)

    print("Probability of class 1 given 'free money':", prob_5)
    print("Probability of class 0 given 'free money':", prob_6)
    print("total:", prob_5 + prob_6)

    print("Try a word that doesn't appear, with and without Laplacian Smoothing")
    prob_7 = n_b(1, 'ubuntu', mails)
    prob_8 = n_b(0, 'ubuntu', mails)
    prob_9= n_b(1, 'ubuntu', mails, smoothing=True)
    prob_10 = n_b(0, 'ubuntu', mails, smoothing=True)

    print("No smoothing:")
    print("Probability of class 1 given 'ubuntu':", prob_7)
    print("Probability of class 0 given 'ubuntu':", prob_8)
    print("With Smoothing:")
    print("Probability of class 1 given 'ubuntu':", prob_9)
    print("Probability of class 1 given 'ubuntu':", prob_10)

    