import numpy as np
import convexa as cx
# import convexa_core as cx
a = np.array(range(5))
b = np.zeros(shape=(5))
b[0] = 1
b[1] = 1
# Convolution is commutative; order doesn't have any effect on output
c = np.convolve(b, a)
c_convexa = cx.host_convolve(cx.dArray(a), cx.dArray(b))
c_r = cx.dArray([3])
timing = cx.host_convolve_timing(cx.dArray(a), cx.dArray(b), c_r)


def max_abs_error (reference, implementation) :
    reference_tst = np.array(reference)
    implementation_tst = np.array(implementation)
    return max(np.abs(implementation_tst - reference_tst) )

def test_all(fn_list, expected_result, *args):
    for fn in fn_list:
        result = fn(*args)
        if (max_abs_error(expected_result, result) > 1e-5):
            status = "FAILED"
        else:
            status = "PASSED"
        # print(f"Error for fn {fn.__name__} = {max_abs_error(expected_result, result)}")
        print(f"fn {fn.__name__}: {status}")





if __name__ == "__main__" :
    a = cx.dArray(a)
    b = cx.dArray(b)
    test_all([cx.host_convolve], c, a, b)
    # print(max_abs_error(c, c_convexa))
    # print(f"Timing Test: {timing}, Result: {c_r}")
