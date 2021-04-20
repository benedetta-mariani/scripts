# author: Dominik Krzeminski (dokato)

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss


from dfa import dfa
from generate import power_law_noise

true_exp = float(input('Insert exponent: '))
x = power_law_noise(2**12, true_exp)
scales, fluct, alpha = dfa(x)
print("True scaling exponent: {}".format(true_exp))
print("Estimated DFA exponent: {}".format(alpha))
