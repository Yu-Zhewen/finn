import numpy as np
import os
import warnings
from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class UpsampleNearestNeighbour_Batch(HLSCustomOp):
    """
    Corresponds to finn-hlslib UpsampleNearestNeighbour_Batch function.
    Upsampling is done with the Nearest Neighbour algorithm.
    The layer expects square feature maps for the in and output.
    """

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            # Size of the output feature map
            "OFMDim": ("i", True, 0),
            # Size of the input feature map
            "IFMDim": ("i", True, 0),
            # Amount of channels of the input feature map
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # Batch size
            "numInputVectors": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_exp_cycles(self):
        OFMDim = self.get_nodeattr("OFMDim")
        batch_size = self.get_nodeattr("numInputVectors")
        exp_cycles = OFMDim * OFMDim * batch_size
        return int(exp_cycles)

    def get_normal_input_shape(self):
        IFMDim = self.get_nodeattr("IFMDim")
        num_ch = self.get_nodeattr("NumChannels")
        batch = self.get_nodeattr("numInputVectors")
        ishape = (batch, IFMDim, IFMDim, num_ch)
        return ishape

    def get_normal_output_shape(self):
        OFMDim = self.get_nodeattr("OFMDim")
        num_ch = self.get_nodeattr("NumChannels")
        batch = self.get_nodeattr("numInputVectors")
        oshape = (batch, OFMDim, OFMDim, num_ch)
        return oshape

    def get_folded_input_shape(self):
        normal_ishape = list(self.get_normal_input_shape())
        return tuple(normal_ishape)

    def get_folded_output_shape(self):
        normal_oshape = list(self.get_normal_output_shape())
        return tuple(normal_oshape)

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert (
            ishape == exp_ishape
        ), "Unexpect input shape for UpsampleNearestNeighbour_Batch."
        # implement tensor with correct shape
        values = np.random.randn(*oshape).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        pass

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        ret = DataType[self.get_nodeattr("inputDataType")]
        return ret

    def get_output_datatype(self):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def get_instream_width(self):
        ibits = self.get_input_datatype().bitwidth()
        ifm_ch = self.get_nodeattr("NumChannels")
        return ibits * ifm_ch

    def get_outstream_width(self):
        obits = self.get_output_datatype().bitwidth()
        ifm_ch = self.get_nodeattr("NumChannels")
        return obits * ifm_ch

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "upsample.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

        ifm_ch = self.get_nodeattr("NumChannels")
        self.code_gen_dict["$DEFINES$"] += ["#define IFMChannels {}".format(ifm_ch)]

        ibits = self.get_input_datatype().bitwidth()
        self.code_gen_dict["$DEFINES$"] += ["#define Input_precision {}".format(ibits)]

        idim = self.get_nodeattr("IFMDim")
        self.code_gen_dict["$DEFINES$"] += ["#define IFMDim {}".format(idim)]

        odim = self.get_nodeattr("OFMDim")
        self.code_gen_dict["$DEFINES$"] += ["#define OFMDim {}".format(odim)]

        batch_size = self.get_nodeattr("numInputVectors")
        self.code_gen_dict["$DEFINES$"] += ["#define numReps {}".format(batch_size)]

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
        )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """UpsampleNearestNeighbour_Batch<OFMDim, IFMDim, IFMChannels,
            ap_uint<Input_precision> > (in0, out, numReps);"""
        ]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output.npy" % code_gen_dir
        oshape = self.get_folded_output_shape()
        oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out, %s, "%s");'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                oshape_cpp_str,
                npy_out,
            )
        ]

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0, hls::stream<%s > &out)"
            % (self.onnx_node.name, packed_hls_type, packed_hls_type)
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out")
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()
        folded_oshape = self.get_folded_output_shape()

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert (
            inp.shape == exp_ishape
        ), """Input shape doesn't
        match expected shape (numInputVectors, ImgDim, ImgDim, NumChannels)."""
        export_idt = self.get_input_datatype()

        reshaped_input = inp.reshape(folded_ishape)
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == folded_oshape
            ), "cppsim did not produce expected folded output shape"
            context[node.output[0]] = context[node.output[0]].reshape(*exp_oshape)
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp)
            odt = export_idt
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )
            # load and reshape output
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
        assert (
            context[node.output[0]].shape == exp_oshape
        ), """Output shape doesn't match expected shape
            (1, OutputDim, OutputDim, NumChannels)."""