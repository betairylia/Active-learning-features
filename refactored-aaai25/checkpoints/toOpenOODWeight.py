from collections import OrderedDict
import sys
import os
import torch

inp = sys.argv[1]

w = torch.load(inp)
original_state_dict = w['state_dict']

translated = OrderedDict()
for key in original_state_dict:

    if not key.startswith('net.'):
        continue

    newkey = ''
    if key.startswith('net.0.'):
        newkey = key[6:]
    elif key.startswith('net.1.'):
        newkey = "fc.%s" % key[6:]

    newkey = newkey.replace("downsample", "shortcut")

    translated[newkey] = original_state_dict[key]

_base = os.path.basename(inp)
base, ext = os.path.splitext(_base)
dirc = os.path.dirname(inp)

outp = os.path.join(dirc, "%s.openood%s" % (base, ext))

torch.save(translated, outp)
print("Saved to %s" % outp)

