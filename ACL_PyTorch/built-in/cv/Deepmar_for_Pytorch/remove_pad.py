import onnx

model = onnx.load("Deepmar.onnx")

model.graph.node[174].input[0] = '492'

node_list = ["Pad_173",'Constant_172']
max_idx = len(model.graph.node)
rm_cnt = 0
for i in range(len(model.graph.node)):
    if i < max_idx:
        n = model.graph.node[i - rm_cnt]
        if n.name in node_list:
            print("remove {} total {}".format(n.name, len(model.graph.node)))
            model.graph.node.remove(n)
            max_idx -= 1
            rm_cnt += 1
onnx.checker.check_model(model)
onnx.save(model, "Deepmar_nopad.onnx")