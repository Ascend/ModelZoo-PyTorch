import sys
import onnx

if __name__ == '__main__':
    model = onnx.load(sys.argv[1])
    graph = model.graph
    node = graph.node
    softmax_node_index = []
    del_group = []
    for i in range(len(node)):
        if node[i].op_type == 'Softmax' and node[i].attribute[0].i == 3:
            del_group.append((node[i-1], node[i], node[i+1], i))
    for g in del_group:
        new_input = g[0].input
        new_output = g[2].output
        new_name = g[1].name
        new_index = g[3]
        new_node = onnx.helper.make_node("Softmax", new_input, new_output, new_name, axis=1)
        for n in g[:-1]:
            graph.node.remove(n)
        graph.node.insert(new_index, new_node)
    onnx.save(model, sys.argv[2])
