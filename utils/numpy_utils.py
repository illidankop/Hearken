import numpy as np

class NumpyUtils:
    #origin_data: where to copy the data
    #new_data: what to copy
    #is_shift_left: should new_data be copied to the start/end of the orgin_data
    #is_slow_no_allocate: on True you save GC but lose performance
    #based on https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    def shift_and_copy(origin_data, new_data, is_shift_left, is_slow_no_allocate = False):
        if is_slow_no_allocate == True:
            if origin_data.ndim == 1:
                return NumpyUtils.shift_and_copy_1D_array_slow_no_allocate(origin_data, new_data, is_shift_left)
            else:
                return NumpyUtils.shift_and_copy_2D_array_slow_no_allocate(origin_data, new_data, is_shift_left)
        else:
            if origin_data.ndim == 1:
                return NumpyUtils.shift_and_copy_1D_array_fast_with_allocate(origin_data, new_data, is_shift_left)
            else:
                return NumpyUtils.shift_and_copy_2D_array_fast_with_allocate(origin_data, new_data, is_shift_left)


    def shift_and_copy_1D_array_slow_no_allocate(origin_data, new_data, is_shift_left):
        num = len(new_data) * -1 if is_shift_left == True else len(new_data)
        if num >= 0:
            return np.concatenate((np.full(num, new_data), origin_data[:-num]))
        else:
            return np.concatenate((origin_data[-num:], np.full(-num, new_data)))


    def shift_and_copy_2D_array_slow_no_allocate(origin_data, new_data, is_shift_left):
        num = new_data.shape[1] * -1 if is_shift_left == True else new_data.shape[1]
        if num >= 0:
            return np.concatenate((np.full((new_data.shape[0],new_data.shape[1]), new_data), origin_data[:,:-num]), axis=1)
        else:           
            return np.concatenate((origin_data[:,-num:], np.full((new_data.shape[0],new_data.shape[1]), new_data)), axis=1)            


    def shift_and_copy_1D_array_fast_with_allocate(origin_data, new_data, is_shift_left):
        result = np.empty_like(origin_data)
        num = len(new_data) * -1 if is_shift_left == True else len(new_data)
        if num > 0:
            result[:num] = new_data
            result[num:] = origin_data[:-num]
        elif num < 0:
            result[num:] = new_data
            result[:num] = origin_data[-num:]
        else:
            result[:] = origin_data
        return result


    def shift_and_copy_2D_array_fast_with_allocate(origin_data, new_data, is_shift_left):    
        result = np.empty_like(origin_data)
        num = new_data.shape[1] * -1 if is_shift_left == True else new_data.shape[1]
        if num > 0:
            result[:,:num] = new_data
            result[:,num:] = origin_data[:,:-num]
        elif num < 0:
            result[:,num:] = new_data
            result[:,:num] = origin_data[:,-num:]
        else:
            result[:] = origin_data
        return result