#include <stdlib.h>
#include "config.hpp"
#include "device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk.hpp"
#include "element_wise_operation.hpp"
#include "device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv2d_bwd_data_instance {

using DataType = int8_t;
using AccType  = int32_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
static constexpr auto ConvBwdDataDefault =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization_t::Default;

static constexpr auto ConvBwdDataFilter1x1Stride1Pad0 =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization_t::Filter1x1Stride1Pad0;

// Compilation parameters for in[n, hi, wi, c] * wei[k, y, x, c] = out[n, ho, wo, k]
using device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_int8_instances = std::tuple<
    // clang-format off
        //######################################################################|    InData|   WeiData|   OutData|    AccData|          In|         Wei|         Out|        ConvBackward| Block|  MPer|  NPer| K0Per|  K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|
        //######################################################################|      Type|      Type|      Type|       Type| Elementwise| Elementwise| Elementwise|                Data|  Size| Block| Block| Block|    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|
        //######################################################################|          |          |          |           |   Operation|   Operation|   Operation|      Specialization|      |      |      |      |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|
        //######################################################################|          |          |          |           |            |            |            |                    |      |      |      |      |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,   256,   256,   128,     4,  16,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,   256,   128,   256,     4,  16,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,   128,   128,   128,     4,  16,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,   256,   128,   128,     4,  16,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,   128,   128,    64,     4,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,   128,    64,   128,     4,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,    64,    64,    64,     4,  16,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 16, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,   256,   128,    64,     4,  16,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              1,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,   256,    64,   128,     4,  16,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,   128,   128,    32,     4,  16,   32,   32,    2,    1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              1,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,   128,    32,   128,     4,  16,   32,   32,    1,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,    64,    64,    32,     4,  16,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 16, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataDefault,    64,    32,    64,     4,  16,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 16, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              16,      true,               7,               1>
    // clang-format on
    >;

using device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_1x1_s1_p0_int8_instances =
    std::tuple<
        // clang-format off
        //#####################################################################|    InData|   WeiData|   OutData|    AccData|          In|         Wei|         Out|                     ConvBackward| Block|  MPer|  NPer| K0Per|  K1| MPer| NPer| MXdl| NXdl|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|
        //#####################################################################|      Type|      Type|      Type|       Type| Elementwise| Elementwise| Elementwise|                             Data|  Size| Block| Block| Block|    |  XDL|  XDL|  Per|  Per|    ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|
        //#####################################################################|          |          |          |           |   Operation|   Operation|   Operation|                   Specialization|      |      |      |      |    |     |     | Wave| Wave|  Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|
        //#####################################################################|          |          |          |           |            |            |            |                                 |      |      |      |      |    |     |     |     |     |                 |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,   256,   256,   128,     4,  16,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,   256,   128,   256,     4,  16,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,   128,   128,   128,     4,  16,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,   256,   128,   128,     4,  16,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,   128,   128,    64,     4,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,   128,    64,   128,     4,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,    64,    64,    64,     4,  16,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 16, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,   256,   128,    64,     4,  16,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              1,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,   256,    64,   128,     4,  16,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,   128,   128,    32,     4,  16,   32,   32,    2,    1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              1,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,   128,    32,   128,     4,  16,   32,   32,    1,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,    64,    64,    32,     4,  16,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 16, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              16,      true,               7,               1>,
        DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  DataType,  DataType,  DataType,    AccType, PassThrough, PassThrough, PassThrough,  ConvBwdDataFilter1x1Stride1Pad0,    64,    32,    64,     4,  16,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,           2,              16,              16,      true,     S<4, 16, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              16,      true,               7,               1>
        // clang-format on
        >;

template <>
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_instances(
    std::vector<DeviceConvBwdDataPtr<PassThrough, PassThrough, PassThrough>>& instances, DataType)
{
    add_device_operation_instances(instances,
                                   device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_int8_instances{});
    add_device_operation_instances(
        instances, device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_1x1_s1_p0_int8_instances{});
}

} // namespace device_conv2d_bwd_data_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
