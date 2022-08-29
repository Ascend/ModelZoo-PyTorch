#!/usr/bin/env python
# coding=utf-8

import sys
import numpy as np
from magiconnx import OnnxGraph

def reshape_axis(graph: OnnxGraph, old_shape: np.int64, new_shape: np.int64) -> None:
    """
    modify 'Reshape' operator shape
    """
    reshape_nodes = graph.get_nodes('Reshape')
    if reshape_nodes is None or len(reshape_nodes) is 0:
        print('There is no Reshape.')
    for node in reshape_nodes:
        input = graph[node.inputs[1]]
        if input is None:
            continue
        newshape = input.value.copy()
        for i in range(0, len(input.value)):
            if input.value[i] == old_shape:
                newshape[i] = new_shape
        input.value = newshape

def add_concat_tensor(graph: OnnxGraph, axis_size: np.int64, axis: np.int64) -> None:
    """
    modify 'Concat' operator, add one tensor to input
    """
    node_name = 'Concat_24'
    initializer_name = '166'

    tensor_shape = list(graph[graph[node_name].inputs[0]].value.shape)
    tensor_shape[axis] = axis_size
    tensor = np.zeros(tuple(tensor_shape), dtype=np.float32)
    graph.add_initializer(initializer_name, tensor)
    graph[node_name].inputs.append(initializer_name)

def del_layernorm_transdata(graph: OnnxGraph) -> bool:
    """
    delete 'Transdata' operator before and after 'LayerNorm' operator
    """
    node = graph['pos_embed']
    if node is None:
        print('pos_embed not exist in graph.')
        return False
    arr = node.value
    if len(arr.shape) < 2:
        print('pos_embed array shape less than 2.')
        return False
    newshape = list(arr.shape) # convert tuple to list
    min_shape = 16
    if (arr.shape[-1] % min_shape) is not 0:
        newshape[-1] = (arr.shape[-1] // min_shape + 1) * min_shape
        reshape_axis(graph, arr.shape[-1], newshape[-1])
        add_concat_tensor(graph, newshape[-1] - arr.shape[-1], -1)
    if (arr.shape[-2] % min_shape) is not 0:
        newshape[-2] = (arr.shape[-2] // min_shape + 1) * min_shape
        reshape_axis(graph, arr.shape[-2], newshape[-2])
        add_concat_tensor(graph, newshape[-2] - arr.shape[-2], -2)
    newarr = np.zeros(tuple(newshape), dtype=arr.dtype)
    if len(arr.shape) is 3:
        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                for k in range(0, arr.shape[2]):
                    newarr[i][j][k] = arr[i][j][k]
        node.value = newarr
    return True

def del_matmul_transpose(graph: OnnxGraph) -> bool:
    """
    delete 'Transpose' operator before 'MatMul' operator by add attribute 'transA' or 'transB'
    """
    matmul_nodes = graph.get_nodes('MatMul')
    transpose_nodes = graph.get_nodes('Transpose')
    for matmul in matmul_nodes:
        for transpose in transpose_nodes:
            if matmul.inputs[0] == transpose.outputs[0]:
                matmul.attrs.setdefault('transA')
                matmul['transA'] = 1
                graph.remove(transpose.name)
                break
            if matmul.inputs[1] == transpose.outputs[0]:
                matmul.attrs.setdefault('transB')
                matmul['transB'] = 1
                graph.remove(transpose.name)
                break
    return True

def improve_model(path: str, new_path: str) -> None:
    print('onnx source path {}, dest path {}'.format(path, new_path))
    graph = OnnxGraph.parse(path)
    if graph is None:
        print('onnx model not exist.')
        return None
    ret = del_layernorm_transdata(graph)
    if not ret:
        print('delete layernorm transdata failed.')
        return None
    graph.save(new_path)
    print('improve onnx model succeed.')


if __name__ == '__main__':
    if len(sys.argv) is not 3:
        print('only need 2 params, include onnx source path and dest path.')
    else:
        improve_model(sys.argv[1], sys.argv[2])
