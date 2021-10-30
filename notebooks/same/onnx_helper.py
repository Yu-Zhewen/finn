import numpy as np
import numpy as np
import brevitas
import torch
import onnx
from finn.custom_op.registry import getCustomOp

#copy from fpgaConvNet optimiser
def add_input_from_initializer(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.input}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.input.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)


    return add_const_value_infos_to_graph(model.graph)


def add_weight_quantization_annotation(model, weight_width=16):
    for initializer in model.graph.initializer:
        tensor_name = initializer.name
        import brevitas
        import torch
        datatype = brevitas.export.onnx.finn.utils.finn_datatype(torch.tensor(weight_width), True)
    
        qa = onnx.TensorAnnotation()
        dt = onnx.StringStringEntryProto()
        dt.key = "finn_datatype"
        dt.value = datatype
        qa.tensor_name = tensor_name
        qa.quant_parameter_tensor_names.append(dt)
        model.graph.quantization_annotation.append(qa)
        
        from onnx import numpy_helper
        W = numpy_helper.to_array(initializer).copy()
        W *= 2 ** (weight_width-1)
        W = W.astype(np.int32)
        initializer.CopyFrom(numpy_helper.from_array(W, initializer.name))


# https://github.com/Xilinx/brevitas/blob/b4390dbe10b663489e71121b9232531a26a8ac36/src/brevitas/export/onnx/finn/handler/act.py
def add_activation_quantization_multithreshold(model, activation_width=16):
    
    def _get_thresholds(data_width, is_signed):
        num_distinct_values = 2 ** data_width
        num_thresholds = num_distinct_values - 1
        step = 1 / (2 ** (data_width-1)) if is_signed else 1 / (2 ** (data_width)-1)
        half_step = step / 2.0
        
        min_value = -1.0 if is_signed else 0.0

        min_threshold = min_value + half_step
        thresholds= []
        for t in range(num_thresholds):
            thresholds.append(min_threshold + step * t) 
            
        thresholds = np.array(thresholds,dtype=np.float32)
        
        return thresholds
        
    
    #quantise inputs
    node_ind = 0
    n = model.graph.node[node_ind]
    node_input = n.input[0]
    channels = model.get_tensor_shape(n.input[0])[1]
    thresholds = _get_thresholds(activation_width, True)
    multi_thresholds = np.repeat([thresholds], channels, axis=0)
    
    threshold_value_info = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), onnx.TensorProto.FLOAT, multi_thresholds.shape)
    ti = model.graph.input.add()
    ti.name = threshold_value_info.name
    model.set_initializer(threshold_value_info.name,multi_thresholds)
    new_input_value1_info = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), onnx.TensorProto.FLOAT, model.get_tensor_shape(n.input[0]))
    model.graph.value_info.append(new_input_value1_info)
    multi_threshold_node = onnx.helper.make_node("MultiThreshold", [node_input, threshold_value_info.name], [new_input_value1_info.name], domain="finn.custom_op.general")
    multi_threshold_node.name = "MultiThreshold" + str(node_ind)
    datatype = brevitas.export.onnx.finn.utils.finn_datatype(torch.tensor(activation_width), True)
    instr = getCustomOp(multi_threshold_node)
    instr.set_nodeattr("out_dtype", datatype)
    
    bias_value_info = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), onnx.TensorProto.FLOAT, [1])
    bi = model.graph.input.add()
    bi.name = bias_value_info.name
    model.set_initializer(bi.name,np.array([-(2 ** (activation_width-1))],dtype=np.float32))
    new_input_value2_info = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), onnx.TensorProto.FLOAT, model.get_tensor_shape(n.input[0]))
    model.graph.value_info.append(new_input_value2_info)
    bias_node = onnx.helper.make_node("Add", [new_input_value1_info.name, bias_value_info.name], [new_input_value2_info.name])

    scale_value_info = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), onnx.TensorProto.FLOAT, [1])
    si = model.graph.input.add()
    si.name = scale_value_info.name
    model.set_initializer(si.name,np.array([1 / (2 ** (activation_width-1))],dtype=np.float32))
    new_input_value3_info = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), onnx.TensorProto.FLOAT, model.get_tensor_shape(n.input[0]))
    model.graph.value_info.append(new_input_value3_info)
    n.input[0] = new_input_value3_info.name
    scale_node = onnx.helper.make_node("Mul", [new_input_value2_info.name, scale_value_info.name], [new_input_value3_info.name])

    model.graph.node.insert(node_ind, scale_node)
    model.graph.node.insert(node_ind, bias_node)        
    model.graph.node.insert(node_ind, multi_threshold_node)
    
                                           
    #replace relu with multithresholds
    for n in model.graph.node:
        node_ind += 1
                                            
        if n.op_type == "Relu":
            node_input = n.input[0]
            node_output = n.output[0]
            channels = model.get_tensor_shape(n.input[0])[1]
            thresholds = _get_thresholds(activation_width, False)
            multi_thresholds = np.repeat([thresholds], channels, axis=0)
                                            
            threshold_value_info = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), onnx.TensorProto.FLOAT, multi_thresholds.shape)
            ti = model.graph.input.add()
            ti.name = threshold_value_info.name
            model.set_initializer(threshold_value_info.name,multi_thresholds)
            new_input_value1_info = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), onnx.TensorProto.FLOAT, model.get_tensor_shape(n.input[0]))
            model.graph.value_info.append(new_input_value1_info)
            multi_threshold_node = onnx.helper.make_node("MultiThreshold", [node_input, threshold_value_info.name], [new_input_value1_info.name], domain="finn.custom_op.general")
            multi_threshold_node.name = "MultiThreshold" + str(node_ind)
            datatype = brevitas.export.onnx.finn.utils.finn_datatype(torch.tensor(activation_width), False)
            instr = getCustomOp(multi_threshold_node)
            instr.set_nodeattr("out_dtype", datatype)
    
            scale_value_info = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), onnx.TensorProto.FLOAT, [1])
            si = model.graph.input.add()
            si.name = scale_value_info.name
            model.set_initializer(si.name,np.array([1 / (2 ** (activation_width)-1)],dtype=np.float32))
            scale_node = onnx.helper.make_node("Mul", [new_input_value1_info.name, scale_value_info.name], [node_output])
            
            model.graph.node.remove(n)
            model.graph.node.insert(node_ind, scale_node)        
            model.graph.node.insert(node_ind, multi_threshold_node)

def remove_redundant_nodes(model):
    for n in model.graph.node:
        if n.op_type in ["Softmax"]:
            from finn.transformation.streamline.remove import _remove_node_and_rewire
            _remove_node_and_rewire(model, n)
    