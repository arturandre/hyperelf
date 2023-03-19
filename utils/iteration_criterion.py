import torch
import torch.nn as nn

from abc import abstractmethod

def shannon_entropy(y, from_logits=True):
    with torch.no_grad():
        if from_logits:
            probs = nn.functional.softmax(y, dim=1)
        else:
            probs = y
        #aux = nn.functional.softmax(y, dim=1)
        base = torch.zeros_like(probs)
        base += y.shape[-1]
        base = torch.log10(base)
        aux = torch.log10(probs+torch.finfo(torch.float32).eps)
        aux = aux/base
        shannon_entropy = -torch.sum(probs*aux)/aux.shape[0] # Batch size
    return shannon_entropy



class IterationCriterion():
    def __init__(self) -> None:
        pass

    def correctness_criterion(self, y_probs, gt):
        """
        If the predictions 'y_probs' are correct,
        according to the 'gt' parameter, then
        it returns true.
        """
        #if gt is None:
        #    raise Exception("Correctness criterion can only be called when a self.gt is not None.")
        pred = torch.argmax(y_probs, dim=1)
        correct = torch.equal(pred, gt)
        return correct

    @abstractmethod
    def custom_criterion(self, y_probs, **kwargs):
        pass

    def __call__(self, y_probs, **kwargs):
        is_correct = True
        if 'gt' in kwargs:
            if kwargs['gt'] is not None:
                is_correct = self.correctness_criterion(y_probs, kwargs['gt'])
            del kwargs['gt']
        return is_correct and self.custom_criterion(y_probs, **kwargs)

class EntropyCriterion(IterationCriterion):
    def __init__(self, threshold=None):
        self.threshold = threshold

    def custom_criterion(self, y_probs, threshold=None):
        """
        Returns false is the entropy of the 
        target vector 'y' is larger
        than a given 'threshold'.

        y: probabilities vector (can be a batch).
        threshold: a value in the range [0,1].
        """
        if threshold is None:
            threshold = self.threshold
            if threshold is None:
                threshold = 0.1
        return shannon_entropy(y_probs) < threshold