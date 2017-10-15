##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__all__ = ['view_each', 'multi_each', 'sum_each', 'upsample']

def view_each(x, size):
    y = []
    for i in range(len(x)):
        y.append(x[i].view(size))
    return y

def multi_each(a, b):
    y = []
    for i in range(len(a)):
        y.append(a[i] * b[i])
    return y

def sum_each(x, y):
    assert(len(x)==len(y))
    z = []
    for i in range(len(x)):
        z.append(x[i]+y[i])
    return z


def upsample(input, size=None, scale_factor=None, mode='nearest'):
    if isinstance(input, Variable):
        return F.upsample(input, size=size, scale_factor=scale_factor,
                          mode=mode)
    elif isinstance(input, tuple) or isinstance(input, list):
        lock = threading.Lock()
        results = {}
        def _worker(i, x):
            try:
                with torch.cuda.device_of(x):
                    result =  F.upsample(x, size=size, \
                        scale_factor=scale_factor,mode=mode)
                with lock:
                    results[i] = result
            except Exception as e:
                with lock:
                    resutls[i] = e 
        # multi-threading for different gpu
        threads = [threading.Thread(target=_worker,
                                    args=(i, x),
                                    )
                   for i, (x) in enumerate(input)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join() 
        # gather the results
        def _list_gather(x):
            y = []
            for i in range(len(x)):
                xi = x[i]
                if isinstance(xi, Exception):
                    raise xi
                y.append(xi)
            return y
        outputs = _list_gather(results)
        return outputs

    else:
        raise RuntimeError('unknown input type')
