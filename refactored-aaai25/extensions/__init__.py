from .commons import *
from .seen_class_acc import *
from .exact_ntk import *

CallbackDict = {
    "dataset_vis": DatasetVisualizer,
    "logits_stats": LogitsStatstics,
    "seen_class_acc": SeenClassesAccuracy,
    "exact_ntk_inf": ExactNTKComputation,
}

def get_callbacks(args):
    
    callbacks = []
    resolved = vars(args)

    for k in CallbackDict.keys():
        if k in resolved and resolved[k] == True:
            callbacks.append(CallbackDict[k](args))
    
    return callbacks
