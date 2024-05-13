from typing import List, Tuple


def get_confusion_matrix(
    y_true: List[int], y_pred: List[int], num_classes: int,
) -> List[List[int]]:
    """
    Generate a confusion matrix in a form of a list of lists. 

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values
    :param num_classes: number of supported classes

    :return: confusion matrix
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Invalid input shapes!")

    conf_matrix = [[0] * num_classes for _ in range(num_classes)]

    for actual, predicted in zip(y_true, y_pred):
        if actual >= num_classes or predicted >= num_classes:
            raise ValueError("Invalid prediction classes!")

        conf_matrix[actual][predicted] += 1

    return conf_matrix


def get_quality_factors(
    y_true: List[int],
    y_pred: List[int],
) -> Tuple[int, int, int, int]:
    """
    Calculate True Negative, False Positive, False Negative and True Positive 
    metrics basing on the ground truth and predicted lists.

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: a tuple of TN, FP, FN, TP
    """
    TN = FP = FN = TP = 0
    for true, pred in zip(y_true, y_pred):
        if true == 0:
            if pred == 0:
                TN += 1

            else:
                FP += 1

        elif true == 1:
            if pred == 1:
                TP += 1

            else:
                FN += 1

    return TN, FP, FN, TP


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the accuracy for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: accuracy score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)
    sum_correct = TP + TN
    total_true = len(y_true)

    if total_true > 0:
        return sum_correct / total_true

    else:
        return 0


def precision_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the precision for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: precision score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)
    TP_FP = TP + FP

    if TP_FP > 0:
        return TP / TP_FP

    else:
        return 0


def recall_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the recall for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: recall score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)
    TP_FN = TP + FN

    if TP_FN > 0:
        return TP / TP_FN

    else:
        return 0


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the F1-score for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: F1-score
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    sum_prec_rec = precision + recall
    mul_prec_rec = precision * recall

    if sum_prec_rec > 0:
        return 2 * mul_prec_rec / sum_prec_rec

    else:
        return 0
