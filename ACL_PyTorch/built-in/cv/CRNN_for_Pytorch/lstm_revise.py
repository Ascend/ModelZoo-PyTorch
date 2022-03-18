import os
import numpy as np
import torch
import onnx
import copy
from onnx import numpy_helper
G_ONNX_OPSET_VER = 11
'''
def onnx_LSTM(batch, seq_len, input_size, hidden_size, num_layers, bidirectional, work_dir):
    mod = torch.nn.LSTM(input_size, hidden_size, num_layers,bidirectional=bidirectional)
    input = torch.randn(seq_len, batch, input_size)#(seq_len, batch, input_size)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    print(mod.weight_ih_l0.size())
    print(mod.bias_hh_l0.size())
    print(mod.bias_ih_l0.size())
    output, _ = mod(input)
    onnx_name = os.path.join(work_dir, "LSTM.onnx")
    torch.onnx.export(mod, (input), f=onnx_name,
                  opset_version=G_ONNX_OPSET_VER)
    return output
'''

def GetNodeIndex(graph, node_name):
    index = 0
    for i in range(len(graph.node)):
        if graph.node[i].name == node_name:
            index = i
            break
    return index
    
def modify1(src_path,save_path):
    model = onnx.load(src_path)
    new_model = onnx.ModelProto()
    for init in model.graph.initializer:
        if init.name == "441": #95
            tmp1 = numpy_helper.to_array(init)
        if init.name == "442": #96
            tmp2 = numpy_helper.to_array(init)
        if init.name == "443": #97
            tmp3 = numpy_helper.to_array(init)
    remove_weight = []
    for init in model.graph.initializer:
        if init.name == "442": #96
            remove_weight.append(init)
        if init.name == "441": #95
            remove_weight.append(init)
        if init.name == "443": #97
            remove_weight.append(init)
    for i in remove_weight:
        model.graph.initializer.remove(i)
    tmp = np.concatenate((tmp1,tmp2),axis=-1)
    tmp_shape = tmp.shape
    tmp = tmp.reshape(4,tmp_shape[-2]//4,tmp_shape[-1]).tolist()
    #print(tmp[0])
    weight = np.array(tmp[0] + tmp[3] + tmp[2] + tmp[1]).reshape(tmp_shape[-2],tmp_shape[-1]).transpose(1,0)
    init_tmp = numpy_helper.from_array(weight.astype(np.float32),name="441") #95
    model.graph.initializer.append(init_tmp)
    bais_shape = [tmp3.shape[-1]]
    tmp3 = tmp3.reshape(2,4,bais_shape[-1]//8).tolist()
    bais1 = np.array(tmp3[0][0] + tmp3[0][3] + tmp3[0][2] + tmp3[0][1]).reshape(bais_shape[-1]//2)
    bais2 = np.array(tmp3[1][0] + tmp3[1][3] + tmp3[1][2] + tmp3[1][1]).reshape(bais_shape[-1]//2)
    bais = bais1 + bais2
    init_bais = numpy_helper.from_array(bais.astype(np.float32),name="443") #97
    model.graph.initializer.append(init_bais)
    for idx,node in enumerate(model.graph.node):
        if node.name == "LSTM_29":
            print(model.graph.node[idx].input)
            #model.graph.node[idx].input[1] = ''
            #model.graph.node[idx].input[2] = ''
            model.graph.node[idx].input.remove('442') #96
            model.graph.node[idx].input.remove('82') #14
            model.graph.node[idx].input.remove('82') #14
            model.graph.node[idx].input.remove('')   #''
            model.graph.node[idx].input[1] = '441'    #95
            model.graph.node[idx].input[2] = '443'    #97
            #model.graph.node[idx].input[3] = '87'
    
    #去除Squeeze
    remove_list = []
    for idx,node in enumerate(model.graph.node):
        if node.name in {"Shape_23", "Gather_25", "Unsqueeze_26","Concat_27","ConstantOfShape_28","Constant_24","Squeeze_30"}:
            remove_list.append(node)
    
    for node in remove_list:
        model.graph.node.remove(node)
    

    
    model.graph.node[GetNodeIndex(model.graph,'Concat_49')].input[0] = '140'
    onnx.save(model,save_path)


def modify2(src_path,save_path):
    model = onnx.load(src_path)
    new_model = onnx.ModelProto()
    for init in model.graph.initializer:
        if init.name == "463": #95
            tmp1 = numpy_helper.to_array(init)
        if init.name == "464": #96
            tmp2 = numpy_helper.to_array(init)
        if init.name == "465": #97
            tmp3 = numpy_helper.to_array(init)
    remove_weight = []
    for init in model.graph.initializer:
        if init.name == "464": #96
            remove_weight.append(init)
        if init.name == "463": #95
            remove_weight.append(init)
        if init.name == "465": #97
            remove_weight.append(init)
    for i in remove_weight:
        model.graph.initializer.remove(i)
    tmp = np.concatenate((tmp1,tmp2),axis=-1)
    tmp_shape = tmp.shape
    tmp = tmp.reshape(4,tmp_shape[-2]//4,tmp_shape[-1]).tolist()
    #print(tmp[0])
    weight = np.array(tmp[0] + tmp[3] + tmp[2] + tmp[1]).reshape(tmp_shape[-2],tmp_shape[-1]).transpose(1,0)
    init_tmp = numpy_helper.from_array(weight.astype(np.float32),name="463") #95
    model.graph.initializer.append(init_tmp)
    bais_shape = [tmp3.shape[-1]]
    tmp3 = tmp3.reshape(2,4,bais_shape[-1]//8).tolist()
    bais1 = np.array(tmp3[0][0] + tmp3[0][3] + tmp3[0][2] + tmp3[0][1]).reshape(bais_shape[-1]//2)
    bais2 = np.array(tmp3[1][0] + tmp3[1][3] + tmp3[1][2] + tmp3[1][1]).reshape(bais_shape[-1]//2)
    bais = bais1 + bais2
    init_bais = numpy_helper.from_array(bais.astype(np.float32),name="465") #97
    model.graph.initializer.append(init_bais)
    for idx,node in enumerate(model.graph.node):
        if node.name == "LSTM_42":
            print(model.graph.node[idx].input)
            #model.graph.node[idx].input[1] = ''
            #model.graph.node[idx].input[2] = ''
            model.graph.node[idx].input.remove('464') #96
            model.graph.node[idx].input.remove('158') #14
            model.graph.node[idx].input.remove('158') #14
            model.graph.node[idx].input.remove('')   #''
            model.graph.node[idx].input[1] = '463'    #95
            model.graph.node[idx].input[2] = '465'    #97
            #model.graph.node[idx].input[3] = '87'
    
    remove_list = []
    for idx,node in enumerate(model.graph.node):
        if node.name in {"Shape_36", "Gather_38", "Unsqueeze_39","Concat_40","ConstantOfShape_41","Constant_37","Squeeze_43"}:
            remove_list.append(node)
    
    for node in remove_list:
        model.graph.node.remove(node)
    
    model.graph.node[GetNodeIndex(model.graph,'Slice_48')].input[0] = '216'
    
    onnx.save(model,save_path)
    
def modify3(src_path,save_path):
    model = onnx.load(src_path)
    new_model = onnx.ModelProto()
    for init in model.graph.initializer:
        if init.name == "486": #95
            tmp1 = numpy_helper.to_array(init)
        if init.name == "487": #96
            tmp2 = numpy_helper.to_array(init)
        if init.name == "488": #97
            tmp3 = numpy_helper.to_array(init)
    remove_weight = []
    for init in model.graph.initializer:
        if init.name == "487": #96
            remove_weight.append(init)
        if init.name == "486": #95
            remove_weight.append(init)
        if init.name == "488": #97
            remove_weight.append(init)
    for i in remove_weight:
        model.graph.initializer.remove(i)
    tmp = np.concatenate((tmp1,tmp2),axis=-1)
    tmp_shape = tmp.shape
    tmp = tmp.reshape(4,tmp_shape[-2]//4,tmp_shape[-1]).tolist()
    #print(tmp[0])
    weight = np.array(tmp[0] + tmp[3] + tmp[2] + tmp[1]).reshape(tmp_shape[-2],tmp_shape[-1]).transpose(1,0)
    init_tmp = numpy_helper.from_array(weight.astype(np.float32),name="486") #95
    model.graph.initializer.append(init_tmp)
    bais_shape = [tmp3.shape[-1]]
    tmp3 = tmp3.reshape(2,4,bais_shape[-1]//8).tolist()
    bais1 = np.array(tmp3[0][0] + tmp3[0][3] + tmp3[0][2] + tmp3[0][1]).reshape(bais_shape[-1]//2)
    bais2 = np.array(tmp3[1][0] + tmp3[1][3] + tmp3[1][2] + tmp3[1][1]).reshape(bais_shape[-1]//2)
    bais = bais1 + bais2
    init_bais = numpy_helper.from_array(bais.astype(np.float32),name="488") #97
    model.graph.initializer.append(init_bais)
    for idx,node in enumerate(model.graph.node):
        if node.name == "LSTM_75":
            print(model.graph.node[idx].input)
            #model.graph.node[idx].input[1] = ''
            #model.graph.node[idx].input[2] = ''
            model.graph.node[idx].input.remove('487') #96
            model.graph.node[idx].input.remove('256') #14
            model.graph.node[idx].input.remove('256') #14
            model.graph.node[idx].input.remove('')   #''
            model.graph.node[idx].input[1] = '486'    #95
            model.graph.node[idx].input[2] = '488'    #97
            #model.graph.node[idx].input[3] = '87'
    
    remove_list = []
    for idx,node in enumerate(model.graph.node):
        if node.name in {"Shape_69", "Gather_71", "Unsqueeze_72","Concat_73","ConstantOfShape_74","Constant_70","Squeeze_76"}:
            remove_list.append(node)
    
    for node in remove_list:
        model.graph.node.remove(node)
        
    model.graph.node[GetNodeIndex(model.graph,'Concat_95')].input[0] = '314'
   
    
    onnx.save(model,save_path)
    
def modify4(src_path,save_path):
    model = onnx.load(src_path)
    new_model = onnx.ModelProto()
    for init in model.graph.initializer:
        if init.name == "508": #95
            tmp1 = numpy_helper.to_array(init)
        if init.name == "509": #96
            tmp2 = numpy_helper.to_array(init)
        if init.name == "510": #97
            tmp3 = numpy_helper.to_array(init)
    remove_weight = []
    for init in model.graph.initializer:
        if init.name == "509": #96
            remove_weight.append(init)
        if init.name == "508": #95
            remove_weight.append(init)
        if init.name == "510": #97
            remove_weight.append(init)
    for i in remove_weight:
        model.graph.initializer.remove(i)
    tmp = np.concatenate((tmp1,tmp2),axis=-1)
    tmp_shape = tmp.shape
    tmp = tmp.reshape(4,tmp_shape[-2]//4,tmp_shape[-1]).tolist()
    #print(tmp[0])
    weight = np.array(tmp[0] + tmp[3] + tmp[2] + tmp[1]).reshape(tmp_shape[-2],tmp_shape[-1]).transpose(1,0)
    init_tmp = numpy_helper.from_array(weight.astype(np.float32),name="508") #95
    model.graph.initializer.append(init_tmp)
    bais_shape = [tmp3.shape[-1]]
    tmp3 = tmp3.reshape(2,4,bais_shape[-1]//8).tolist()
    bais1 = np.array(tmp3[0][0] + tmp3[0][3] + tmp3[0][2] + tmp3[0][1]).reshape(bais_shape[-1]//2)
    bais2 = np.array(tmp3[1][0] + tmp3[1][3] + tmp3[1][2] + tmp3[1][1]).reshape(bais_shape[-1]//2)
    bais = bais1 + bais2
    init_bais = numpy_helper.from_array(bais.astype(np.float32),name="510") #97
    model.graph.initializer.append(init_bais)
    for idx,node in enumerate(model.graph.node):
        if node.name == "LSTM_88":
            print(model.graph.node[idx].input)
            #model.graph.node[idx].input[1] = ''
            #model.graph.node[idx].input[2] = ''
            model.graph.node[idx].input.remove('509') #96
            model.graph.node[idx].input.remove('332') #14
            model.graph.node[idx].input.remove('332') #14
            model.graph.node[idx].input.remove('')   #''
            model.graph.node[idx].input[1] = '508'    #95
            model.graph.node[idx].input[2] = '510'    #97
            #model.graph.node[idx].input[3] = '87'
    
    remove_list = []
    for idx,node in enumerate(model.graph.node):
        if node.name in {"Shape_82", "Gather_84", "Unsqueeze_85","Concat_86","ConstantOfShape_87","Constant_83","Squeeze_89"}:
            remove_list.append(node)
    
    for node in remove_list:
        model.graph.node.remove(node)
    model.graph.node[GetNodeIndex(model.graph,'Slice_94')].input[0] = '390'
   
    onnx.save(model,save_path)
  
if __name__ == "__main__":
    work_dir = "./"
    batch = 1
    seq_len = 10
    input_size = 50
    hidden_size = 32
    num_layers = 1
    #onnx_LSTM(batch, seq_len, input_size, hidden_size, num_layers, False, work_dir)
    modify1("./crnn_sim.onnx","./1.onnx")
    modify2("./1.onnx","./2.onnx")
    modify3("./2.onnx","./3.onnx")
    modify4("./3.onnx","./crnn_revised.onnx")
    os.remove("1.onnx")
    os.remove("2.onnx")
    os.remove("3.onnx")
    print('Done')
