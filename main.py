from test_mails import prob71 as mails
from naive_bayes import n_b
import mail_parser as m_p

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
    print("\nProbability of class 1 given 'money':", prob_3)
    print("Probability of class 0 given 'money':", prob_4)
    print("total:", prob_3 + prob_4)

    print("\nProbability of class 1 given 'free money':", prob_5)
    print("Probability of class 0 given 'free money':", prob_6)
    print("total:", prob_5 + prob_6)

    print("\nTry a word that doesn't appear, with and without Laplacian Smoothing")
    prob_7 = n_b(1, 'ubuntu', mails)
    prob_8 = n_b(0, 'ubuntu', mails)
    prob_9= n_b(1, 'ubuntu', mails, smoothing=True)
    prob_10 = n_b(0, 'ubuntu', mails, smoothing=True)

    print("No smoothing:")
    print("Probability of class 1 given 'ubuntu':", prob_7)
    print("Probability of class 0 given 'ubuntu':", prob_8)
    print("\nWith Smoothing:")
    print("Probability of class 1 given 'ubuntu':", prob_9)
    print("Probability of class 1 given 'ubuntu':", prob_10)
    print("total:", prob_9 + prob_10)

    # test mails with labeled
    mail_labels = open("SPAMTrain.label", errors="ignore")

    train_mails_dataset = []

    for line in mail_labels:
        line_arr = line.split()

        mail_content = m_p.extract_free_string_from_mail("./TRAINING_RES/" + line_arr[1])
        mail_obj = [mail_content, int(line_arr[0])]

        train_mails_dataset.append(mail_obj)

    print("\nDataset examples:")
    d_prob1 = n_b(1, 'money', train_mails_dataset, smoothing=True)
    d_prob2 = n_b(0, 'money', train_mails_dataset, smoothing=True)

    print("Probability of class 1 given 'money':", d_prob1)
    print("Probability of class 0 given 'money':", d_prob2)
    print("total:", d_prob1 + d_prob2)

    d_prob3 = n_b(1, 'free money', train_mails_dataset, smoothing=True)
    d_prob4 = n_b(0, 'free money', train_mails_dataset, smoothing=True)

    print("Probability of class 1 given 'free money':", d_prob3)
    print("Probability of class 0 given 'free money':", d_prob4)
    print("total:", d_prob3 + d_prob4)

    testing_mail = m_p.extract_free_string_from_mail("./TESTING_RES/TEST_00000.eml")

    print("\nDataset Input:")
    d_prob5 = n_b(1, testing_mail, train_mails_dataset, smoothing=True)
    d_prob6 = n_b(0, testing_mail, train_mails_dataset, smoothing=True)

    print("Probability of class 1 given a test mail:", d_prob5)
    print("Probability of class 0 given a test mail:", d_prob6)
    print("-->Returns error because of underflow...")

