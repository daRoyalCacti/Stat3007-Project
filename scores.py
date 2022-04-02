from collections import Counter


# actual accuracy
def get_accuracy(p_data, t_data):
    sc = 0
    for i in range(len(p_data)):
        if p_data[i] == t_data[i]:
            sc += 1
    return sc / len(p_data)


# if all the digits where predicted, not just in the right order
def get_accuracy_order(p_data, t_data):
    sc = 0
    for i in range(len(p_data)):
        pred = str(int(p_data[i]))
        tru = str(t_data[i])
        # https://codereview.stackexchange.com/questions/140807/check-if-one-string-is-a-permutation-of-another-using-python
        if len(pred) == len(tru) and Counter(pred) == Counter(tru):
            sc += 1

    return sc / len(p_data)


# if at least 1 digit is correct
def get_accuracy_one(p_data, t_data):
    sc = 0
    for i in range(len(p_data)):
        pred = str(int(p_data[i]))
        tru = str(t_data[i])

        unique_pred = list(set(pred))
        unique_tru = list(set(tru))

        for u in unique_pred:
            if u in unique_tru:
                sc += 1
                break

    return sc / len(p_data)
