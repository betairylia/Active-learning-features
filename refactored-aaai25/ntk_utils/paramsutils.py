import torch

def unnamed_tuple_to_named(paramsKey, paramsValue):
    return {k: v for k, v in zip(paramsKey.keys(), paramsValue)}

def to_unnamed(params):
    return tuple(params.values())

def params_zip(paramsA, paramsB):
    for k in paramsA:
        yield (k, paramsA[k], paramsB[k])

def params_zeros_like(params):
    return {k: torch.zeros_like(p) for k, p in params.items()}

def params_add(paramsA, paramsB):
    return {k: a + b for k, a, b in params_zip(paramsA, paramsB)}

def params_substract(paramsA, paramsB):
    return {k: a - b for k, a, b in params_zip(paramsA, paramsB)}

def params_multiply(paramsA, paramsB):
    return {k: a * b for k, a, b in params_zip(paramsA, paramsB)}

def params_safe_divide(paramsA, paramsB, margin):
    return {k: a / (torch.sign(b) * (margin + torch.abs(b))) for k, a, b in params_zip(paramsA, paramsB)}

def params_safe_divide_unsigned(paramsA, paramsB, margin):
    return {k: a / (margin + torch.abs(b)) for k, a, b in params_zip(paramsA, paramsB)}

def params_scale(params, scale):
    return {k: p * scale for k, p in params.items()}

def params_pow(params, a):
    return {k: torch.pow(p, a) for k, p in params.items()}

def params_abs(params):
    return {k: torch.abs(p) for k, p in params.items()}

def params_randn_like(params):
    return {k: torch.randn_like(p) for k, p in params.items()}

def params_detach(params):
    return {k: p.detach() for k, p in params.items()}

def params_sum(params):

    result = 0
    for k, v in params.items():
        result += v.sum()
    
    return result

def params_l0norm(params):

    result = 0
    for k, v in params.items():
        result += torch.count_nonzero(v)
    
    return result

def params_l1norm(params):

    result = 0
    for k, v in params.items():
        result += v.abs().sum()
    
    return result

def params_l2norm(params):

    result = 0
    for k, v in params.items():
        result += (v ** 2).sum()
    
    return torch.sqrt(result)

def params_l2norm_sq(params):

    result = 0
    for k, v in params.items():
        result += (v ** 2).sum()
    
    return result

def params_apply_dropout(params, p):
    
    # - For reference -
    # LeNet-5 parameters:
    # ipdb> print("\n".join([repr((p[0], p[1].shape)) for p in self.fparams.items()]))
    # ('0.0.0.weight', torch.Size([6, 1, 5, 5]))
    # ('0.0.0.bias', torch.Size([6]))
    # ('0.0.3.weight', torch.Size([16, 6, 5, 5]))
    # ('0.0.3.bias', torch.Size([16]))
    # ('0.2.0.weight', torch.Size([120, 400]))
    # ('0.2.0.bias', torch.Size([120]))
    # ('0.2.2.weight', torch.Size([84, 120]))
    # ('0.2.2.bias', torch.Size([84]))
    # ('1.weight', torch.Size([10, 84]))
    # ('1.bias', torch.Size([10]))

    new_parameters = {}

    for k, v in params.items():

        # Ignore bias, beta, gamma, etc.
        if v.dim() <= 1:
            new_parameters[k] = v
            continue
        
        # Apply dropout to first two dimensions
        # i.e., channel-wise dropout for convolutional layers

        mask = torch.full((v.shape[0], v.shape[1], *[1 for i in range(v.dim() - 2)]), p, device = v.device)
        mask = torch.bernoulli(mask)
        new_parameters[k] = mask * v
    
    return new_parameters

def params_apply_p_minus_1_dropout(params, p):
    return {k: p_theta - theta for k, theta, p_theta in params_zip(params, params_apply_dropout(params, p))}

class ParamsStatsticsRecorder():

    def __init__(self):

        self.E = None
        self.raw_2m = None
        self.count = 0

    def record(self, result_input):
        
        result = params_detach(result_input)

        if self.E is None:
            self.E = result
        else:
            self.E = params_add(self.E, result)

        if self.raw_2m is None:
            self.raw_2m = params_pow(result, 2.0)
        else:
            self.raw_2m = params_add(self.raw_2m, params_pow(result, 2.0))

        self.count += 1

    def mean(self):
        return params_scale(self.E, 1.0 / self.count)
    
    def variance(self):
        # TODO: Floating point number precision problems? log scale?
        return params_substract(
            params_scale(self.raw_2m, 1.0 / self.count),
            params_pow(self.mean(), 2.0))

    def get_raw_2m(self):
        return params_scale(self.raw_2m, 1.0 / self.count)

def softmax_gradient_at_idx(logits, idx):
    scaled_logits = logits - logits.max()
    logits_without_idx = torch.cat([scaled_logits[:idx], scaled_logits[idx + 1:]])
    log_gradient = scaled_logits[idx] + torch.logsumexp(logits_without_idx, -1) - 2 * torch.logsumexp(scaled_logits, -1)
    return torch.exp(log_gradient)
