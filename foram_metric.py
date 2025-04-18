import numpy as np
import pandas as pd
import pandas.api.types

import kaggle_metric_utilities

import sklearn.metrics

from typing import Sequence, Union, Optional


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, labels: Optional[Sequence]=None, pos_label: Union[str, int]=1, average: str='macro', weights_column_name: Optional[str]=None) -> float:
    '''
    Wrapper for https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    Compute the F1 y_pred, also known as balanced F-score or F-measure.

    The F1 score can be interpreted as a harmonic mean of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

    F1 = 2 * (precision * recall) / (precision + recall)

    In the multi-class and multi-label case, this is the average of
    the F1 score of each class with weighting depending on the ``average``
    parameter.

    Parameters
    ----------
    solution : 1d DataFrame, or label indicator array / sparse matrix
    Ground truth (correct) target values.

    submission : 1d DataFrame, or label indicator array / sparse matrix
    Estimated targets as returned by a classifier.

    labels : array-like, default=None
    The set of labels to include when ``average != 'binary'``, and their
    order if ``average is None``. Labels present in the data can be
    excluded, for example to calculate a multiclass average ignoring a
    majority negative class, while labels not present in the data will
    result in 0 components in a macro average. For multilabel targets,
    labels are column indices. By default, all labels in ``y_true`` and
    ``y_pred`` are used in sorted order.

    pos_label : str or int, default=1
    The class to report if ``average='binary'`` and the data is binary.
    If the data are multiclass or multilabel, this will be ignored;
    setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
    scores for that label only.

    average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, default='binary'
    This parameter is required for multiclass/multilabel targets.
    If ``None``, the scores for each class are returned. Otherwise, this
    determines the type of averaging performed on the data:

    ``'binary'``:
    Only report results for the class specified by ``pos_label``.
    This is applicable only if targets (``y_{true,pred}``) are binary.
    ``'micro'``:
    Calculate metrics globally by counting the total true positives,
    false negatives and false positives.
    ``'macro'``:
    Calculate metrics for each label, and find their unweighted
    mean.  This does not take label imbalance into account.
    ``'weighted'``:
    Calculate metrics for each label, and find their average weighted
    by support (the number of true instances for each label). This
    alters 'macro' to account for label imbalance; it can result in an
    F-score that is not between precision and recall.
    ``'samples'``:
    Calculate metrics for each instance, and find their average (only
    meaningful for multilabel classification where this differs from
    `accuracy_score`).

    weights_column_name: optional str, the name of the sample weights column in the solution file.

    <https://en.wikipedia.org/wiki/F1_score>`_.

    Examples
    --------

    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name, average='macro')
    0.26...
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name, average='micro')
    0.33...
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name, average='weighted')
    0.26...
    '''
    # Skip sorting and equality checks for the row_id_column since that should already be handled
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    sample_weight = None
    if weights_column_name:
        if weights_column_name not in solution.columns:
            raise ValueError(f'The solution weights column {weights_column_name} is not found')
        sample_weight = solution.pop(weights_column_name).values
        if not pandas.api.types.is_numeric_dtype(sample_weight):
            raise ParticipantVisibleError('The solution weights are not numeric')

    if not((len(submission.columns) == 1) or (len(submission.columns) == len(solution.columns))):
        raise ParticipantVisibleError(f'Invalid number of submission columns. Found {len(submission.columns)}')

    solution = solution.values
    submission = submission.values

    score_result = kaggle_metric_utilities.safe_call_score(sklearn.metrics.f1_score, solution, submission, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)

    return score_result