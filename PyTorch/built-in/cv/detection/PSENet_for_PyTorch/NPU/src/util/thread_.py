#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#



import threading
from threading import Thread


def get_current_thread():
    return threading.current_thread()


def get_current_thread_name():
    return get_current_thread().getName()


def is_alive(t):
    return t.is_alive()


def create_and_start(name, target, daemon=True):
    t = Thread(target=target)
    t.daemon = True
    t.setName(name)
    t.start()
    return t


class ThreadPool(object):
    def __init__(self, capacity=10):
        import threadpool
        self.num_threads = capacity
        self.pool = threadpool.ThreadPool(10)

    def add(self, fn, args):
        import threadpool
        if type(args) == list:
            args = [(args, None)]
        elif type(args) == dict:
            args = [(None, args)]
        else:
            raise ValueError("Unsuported args,", type(args))
        request = threadpool.makeRequests(fn, args)[0]
        self.pool.putRequest(request, block=False)
        self.pool.poll()

    def join(self):
        self.pool.wait()


class ProcessPool(object):
    """
    Remember that function in function is not supported by multiprocessing.
    """

    def __init__(self, capacity=8):
        from multiprocessing import Pool

        self.capacity = capacity
        self.pool = Pool(capacity)

    def add(self, fn, args):
        self.pool.apply_async(fn, args)

    #         self.pool.poll()
    #         self.pool.poll

    def join(self):
        self.pool.close()
        self.pool.join()
