# Author:Zhang Yuan

import multiprocessing
import timeit
def do_something(x):
    v = pow(x, 2)
    return v

if __name__ == '__main__':

    # a =[]
    # start = timeit.default_timer()
    # for i in range(1, 100000000):
    #     a.append(do_something(i))
    # end = timeit.default_timer()
    # print('single processing time:', str(end-start), 's')
    # print(a[1:10])

    # revise to parallel
    items = [x for x in range(1, 100000000)]
    p = multiprocessing.Pool(8)
    start = timeit.default_timer()
    b = p.map(do_something, items)
    p.close()
    p.join()
    end = timeit.default_timer()
    print('multi processing time:', str(end-start), 's')
    print(b[1:10])
    # print('Return values are all equal ?:', operator.eq(a, b))




import time

def func(x):
    x = (x**2)**2
    return x

lists = list(range(10000000))

start = time.time()
a=[]
for num in lists:
    a.append(func(num))
end = time.time()
print('Serial computing time:\t',end - start)

start = time.time()
# map() 这个函数能够将for循环的串行计算改变成并行计算。
b = map(func,lists)
end = time.time()
print('Parallel Computing time:\t',end - start)

print(a == list(b))

