import numpy as np
from keras.utils import to_categorical
import collections
import string

from keras import backend as K
from os import stat


def textToSequence(text):
    return np.array([ord(c) for c in list(text)])


def fixDia(text):
    transformationTable = {
        "ş": "ş",
        "ţ": "ţ",
    }

    for char in transformationTable.keys():
        text = text.replace(char, transformationTable[char])

    return text


def toTarget(text):
    text = fixDia(text)

    returnable = []
    for char in text.lower():
        if char in ["ă", "ș", "ț"]:
            returnable.append([0])
        elif char in ["î"]:
            returnable.append([1])
        elif char in ["â"]:
            returnable.append([2])
        else:
            returnable.append([3])

    returnable = breakInto(returnable)
    returnable = to_categorical(returnable, 4)

    return returnable


def removeDiacritics(text):
    text = fixDia(text)

    transformationTable = {
        "î": "i",
        "ă": "a",
        "ț": "t",
        "â": "a",
        "ș": "s",
        "Î": "I",
        "Ă": "A",
        "Ț": "T",
        "Â": "A",
        "Ș": "S",
    }

    for char in transformationTable.keys():
        text = text.replace(char, transformationTable[char])

    return text


def breakInto(arr, n=30):
    arr = np.array(arr, dtype=int)

    if len(arr) % n:
        padSize = n - (len(arr) % n)
        pad = np.zeros((1, padSize), dtype=int)
        arr = np.append(arr, pad)

    arr = np.reshape(arr, (int(len(arr) / n), -1))

    return arr


def single_class_accuracy(interesting_class_id):
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)

        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(
            K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(
            K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / \
            K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return fn
