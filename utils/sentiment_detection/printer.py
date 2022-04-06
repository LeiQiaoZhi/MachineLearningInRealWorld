def print_binary_confusion_matrix(matrix):
    [[tp, fp], [fn, tn]] = matrix
    print(f"""             ACTUAL
          | pos | neg |
     -----+-----+-----+
      pos | {tp:>3} | {fp:>3} |
PRED -----+-----+-----+
      neg | {fn:>3} | {tn:>3} |
     -----+-----+-----+
""")


def print_agreement_table(agreement_table):
    print(f"""                   REVIEW
          |  0  |  1  |  2  |  3  |
     -----+-----+-----+-----+-----+
      pos | {agreement_table[0][1]:>3} | {agreement_table[1][1]:>3} | {agreement_table[2][1]:>3} | {agreement_table[3][1]:>3} |
SENT -----+-----+-----+-----+-----+
      neg | {agreement_table[0][-1]:>3} | {agreement_table[1][-1]:>3} | {agreement_table[2][-1]:>3} | {agreement_table[3][-1]:>3} |
     -----+-----+-----+-----+-----+
    """)
