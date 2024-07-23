from .dataset_visualizer import *
from .seen_class_acc import *

CallbackDict = {
    "dataset_vis": DatasetVisualizer,
    "seen_class_acc": SeenClassesAccuracy,
}

def get_callbacks(args):
    
    callbacks = []
    resolved = vars(args)

    for k in CallbackDict.keys():
        if k in resolved and resolved[k] == True:
            callbacks.append(CallbackDict[k]())
    
    return callbacks
