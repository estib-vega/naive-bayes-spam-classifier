# naive-bayes-spam-classifier
Simple python implementation for a SPAM mail classifier


This is my python implementation of as SPAM Filter using naive bayes to
cllasify sample emails.

To init the program simply type into the Terminal (Command Line): python main.py

- It will then start parsing the sample mails, which were based on a school exercise, mostly for debugging.
- It will print the results with little commentary on the differences between them.

Eventually the programm will begin using the sample emails for training and testing (100 emails each).

All the math formulas are on the 'naive_bayes.py' file, and are shortly explained.

The 'mail_parser.py' only extracts the raw text from the emails, for word processing.

# The two main functions for naive bayes are:
- n_b
- n_b_underflow

'n_b' takes a class, a word (or words), a set of mails to train with and the option to 
do Laplacian Smoothing.

'n_b_underflow' takes some text and a set to train with, and returns the most probable class of the input text.

# The main test

The main test consist in using 100 labeled emails for training and then using other 100 labeled emails
for scoring the results.

The scoring method used is the F1-Precission and Recall function, which is then printed at the end of the tests.
