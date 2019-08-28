import numpy as np
cimport numpy as np
#ctypedef np.int_t DTYPE_t
#from libcpp.queue cimport priority_queue
from libcpp.queue cimport priority_queue


def cummedian(np.ndarray arr):
    cdef int vmax = arr.shape[0]
    cdef np.ndarray out = np.zeros((vmax, 2), dtype=arr.dtype)
    cdef priority_queue[double] small_queue = priority_queue[double]()
    cdef priority_queue[double] large_queue = priority_queue[double]()
    cdef long N = arr.shape[0] 
    small_queue.push(arr[0])
    out[0, 0] = arr[0]
    out[0, 1] = arr[0]
    for i in range(1, N):
        if arr[i] <= small_queue.top():
            small_queue.push(arr[i])
        else:
            large_queue.push(-arr[i])
        
        if small_queue.size() > large_queue.size() + 1:
            large_queue.push(-small_queue.top())
            small_queue.pop()
        elif large_queue.size() > small_queue.size() + 1:
            small_queue.push(-large_queue.top())
            large_queue.pop()
        
        if small_queue.size() == large_queue.size():
            out[i, 0] = small_queue.top()
            out[i, 1] = -large_queue.top()
        if small_queue.size() > large_queue.size():
            out[i, 0] = small_queue.top()
            out[i, 1] = small_queue.top()
        if small_queue.size() < large_queue.size():
            out[i, 0] = -large_queue.top()
            out[i, 1] = -large_queue.top()
    return out
