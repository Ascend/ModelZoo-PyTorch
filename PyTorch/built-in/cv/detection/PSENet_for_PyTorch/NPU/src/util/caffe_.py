# encoding=utf-8

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



import util


def get_data(net, name):
    import caffe
    if isinstance(net, caffe._caffe.Solver):
        net = net.net
    return net.blobs[name].data[...]


def get_params(net, name=None):
    import caffe
    if isinstance(net, caffe._caffe.Solver):
        net = net.net
    params = net.params[name]
    p = []
    for param in params:
        p.append(param.data[...])
    return p


def draw_log(log_path, output_names, show=False, save_path=None, from_to=None, smooth=False):
    pattern = "Train net output: word_bbox_loc_loss = "
    log_path = util.io.get_absolute_path(log_path)
    f = open(log_path, 'r')
    iterations = []
    outputs = {}
    plt = util.plt.plt
    for line in f.readlines():
        if util.str.contains(line, 'Iteration') and util.str.contains(line, 'loss = '):
            print(line)
            s = line.split('Iteration')[-1]
            iter_num = util.str.find_all(s, '\d+')[0]
            iter_num = int(iter_num)
            iterations.append(iter_num)

        if util.str.contains(line, "Train net output #"):
            s = util.str.split(line, 'Train net output #\d+\:')[-1]
            s = s.split('(')[0]
            output = util.str.find_all(s, '\d*\.*\d+e*\-*\d*\.*\d*')[-1]
            output = eval(output)
            output = float(output)
            for name in output_names:
                ptr = ' ' + name + ' ='
                if util.str.contains(line, ptr):
                    if name not in outputs:
                        outputs[name] = []
                    print(line)
                    print('\t', iter_num, name, output)
                    outputs[name].append(output)
    if len(outputs) == 0:
        print('No output named:', output_names)
        return
    for name in outputs:
        output = outputs[name]
        if smooth:
            output = util.np.smooth(output)
        start = 0
        end = len(output)

        if from_to is not None:
            start = from_to[0]
            end = from_to[1]
        line_style = util.plt.get_random_line_style()
        plt.plot(iterations[start: end], output[start: end], line_style, label=name)

    plt.legend()

    if save_path is not None:
        util.plt.save_image(save_path)
    if show:
        util.plt.show()
