
def weight_only_ignore_bn(name, shape):
    return '.weight' in name and 'bn' not in name
