from test_mails import prob71 as mails
from naive_bayes import n_b, n_b_underflow
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

    testing_mail = m_p.extract_free_string_from_mail("./TESTING_RES/TRAIN_00100.eml")

    print("\nDataset Input:")
    d_prob5 = n_b(1, testing_mail, train_mails_dataset, smoothing=True)
    d_prob6 = n_b(0, testing_mail, train_mails_dataset, smoothing=True)

    print("Probability of class 1 given a test mail:", d_prob5)
    print("Probability of class 0 given a test mail:", d_prob6)
    print("-->Returns error because of underflow...")
    print("\nImplement log technique")
    d_prob7 = n_b_underflow(testing_mail, train_mails_dataset)
    print("Most probable class:", d_prob7)

    # test in the 99 other mails
    print("\nOther mails:")
    testing_labels = {}
    
    mail_test_labels = open("./SPAMTest.label", errors="ignore")

    for line in mail_test_labels:
        line_arr = line.split()
        testing_labels[line_arr[1]] = int(line_arr[0])

    num_of_errors = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0

    for num in range(101, 140): # 200
        text_num = '%05d' % num
        test_m = m_p.extract_free_string_from_mail("./TESTING_RES/TRAIN_" + text_num + ".eml")
        prob = n_b_underflow(test_m, train_mails_dataset)
        print("Most probable class for TRAIN_" + text_num + ".eml:", prob)
        if prob == testing_labels["TRAIN_" + text_num + ".eml"]:
            print("------------> correct")
            if prob == 1: true_positives += 1
        else:
            print("--> incorrect")
            num_of_errors += 1
            if prob == 1: false_positives += 1
            else: false_negatives += 1

    print("Number of errors:", num_of_errors)
    precission = (true_positives / (true_positives + false_positives))
    print("Precission:", precission)
    recall = (true_positives / (true_positives + false_negatives))
    print("Recall:", recall)
    print("F1:", (2 * precission * recall / (precission + recall)))

    """
        Number of errors: 5
        Precission: 0.8846153846153846
        Recall: 0.92
        F1: 0.9019607843137256
    """