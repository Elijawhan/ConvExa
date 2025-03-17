import numpy as np
import convexa as cx
# import convexa_core as cx
a = np.array(range(5))
b = np.zeros(shape=(5))
b[0] = 1
b[1] = 1
# Convolution is commutative; order doesn't have any effect on output
c = np.convolve(b, a)
c_convexa = cx.host_convolve(a, b)



def max_abs_error (reference, implementation) :
    reference_tst = np.array(reference)
    implementation_tst = np.array(implementation)
    return max(np.abs(implementation_tst - reference_tst) )





if __name__ == "__main__" :
    print(c)
    print(c_convexa)
    print(max_abs_error(c, c_convexa))
