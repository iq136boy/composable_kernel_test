#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_convolution_resize_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp"

template <typename TInWei,
          typename TAcc,
          typename TBias,
          typename TOut,
          typename InLengths,
          typename WeiLengths,
          typename AddLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_convolution_resize_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1(
    const InLengths& in_n_c0_hi_wi_c1_lengths,
    const WeiLengths& wei_k_c0_y_x_c1_lengths,
    const AddLengths& add_n_k0_hox2_wox2_k1_lengths,
    const OutLengths& out_n_k0_ho_wo_k1_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_c0_hi_wi_c1,
    const Tensor<TInWei>& wei_k_c0_y_x_c1,
    const Tensor<TBias>& bias_k0_k1,
    const Tensor<TOut>& add_n_k0_hox2_wox2_k1,
    Tensor<TOut>& add_n_k0_hox2_wox2_k1_out,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    const auto N  = out_n_k0_ho_wo_k1_lengths[I0];
    const auto K0 = out_n_k0_ho_wo_k1_lengths[I1];
    const auto Ho = out_n_k0_ho_wo_k1_lengths[I2];
    const auto Wo = out_n_k0_ho_wo_k1_lengths[I3];
    const auto K1 = out_n_k0_ho_wo_k1_lengths[I4];

    const auto C0 = in_n_c0_hi_wi_c1_lengths[I1];
    const auto Hi = in_n_c0_hi_wi_c1_lengths[I2];
    const auto Wi = in_n_c0_hi_wi_c1_lengths[I3];
    const auto C1 = in_n_c0_hi_wi_c1_lengths[I4];

    const auto K = wei_k_c0_y_x_c1_lengths[I0];
    const auto Y = wei_k_c0_y_x_c1_lengths[I2];
    const auto X = wei_k_c0_y_x_c1_lengths[I3];

    const auto Hox2 = add_n_k0_hox2_wox2_k1_lengths[I2];
    const auto Wox2 = add_n_k0_hox2_wox2_k1_lengths[I3];

    DeviceMem in_n_c0_hi_wi_c1_device_buf(sizeof(TInWei) *
                                          in_n_c0_hi_wi_c1.mDesc.GetElementSpace());
    DeviceMem wei_k_c0_y_x_c1_device_buf(sizeof(TInWei) * wei_k_c0_y_x_c1.mDesc.GetElementSpace());
    DeviceMem bias_k0_k1_device_buf(sizeof(TBias) * bias_k0_k1.mDesc.GetElementSpace());
    DeviceMem add_n_k0_hox2_wox2_k1_device_buf(sizeof(TOut) *
                                               add_n_k0_hox2_wox2_k1.mDesc.GetElementSpace());

    in_n_c0_hi_wi_c1_device_buf.ToDevice(in_n_c0_hi_wi_c1.mData.data());
    wei_k_c0_y_x_c1_device_buf.ToDevice(wei_k_c0_y_x_c1.mData.data());
    bias_k0_k1_device_buf.ToDevice(bias_k0_k1.mData.data());
    add_n_k0_hox2_wox2_k1_device_buf.ToDevice(add_n_k0_hox2_wox2_k1.mData.data());

    GridGemmTuningParameters<256,        // BlockSize
                             C0 * Y * X, // E1
                             C1,         // E2
                             2,          // K2
                             1,          // E0PerBlock
                             K,          // KPerBlock
                             16,         // HoPerBlock
                             64,         // WoPerBlock
                             2,          // E1PerBlock
                             K,          // KPerThread
                             2,          // HoPerThread
                             2,          // WoPerThread
                             1,          // EPerThread
                             Sequence<1,
                                      C0 * Y * X,
                                      1,
                                      K,
                                      C1>, // ABlockTransferBlockSliceLengths_E0_E1_K0_K1_E2
                             Sequence<1,
                                      C0,
                                      1,
                                      K,
                                      1>, // ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2
                             C1,          // ABlockTransferSrcScalarPerVector_E2
                             C1,          // ABlockTransferDstScalarPerVector_E2
                             C1,          // BThreadTransferSrcScalarPerVector_E2
                             K1           // CThreadTransferDstScalarPerVector_K
                             >
        conv_tuning_parameters{};

    const auto in_n_c0_hi_wi_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, C0, Hi, Wi, C1));
    const auto wei_k_c0_y_x_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(K, C0, Y, X, C1));
    const auto resize_n_k0_hx_wx_k1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Hox2, Wox2, K1));
    const auto out_n_k0_ho_wo_k1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1));

    constexpr auto conv_desc = ConvBiasActivResizeDesc<decltype(in_n_c0_hi_wi_c1_desc),
                                                       decltype(wei_k_c0_y_x_c1_desc),
                                                       decltype(out_n_k0_ho_wo_k1_desc),
                                                       decltype(resize_n_k0_hx_wx_k1_desc),
                                                       decltype(conv_strides),
                                                       decltype(conv_dilations),
                                                       decltype(in_left_pads),
                                                       decltype(in_right_pads)>{};

    constexpr auto conv_driver =
        DriverDynamicConvolutionForwardImplicitGemmDlops_v5r1_nc0hwc1_kc0yxc1_nk0hwk1_resize<
            TInWei,
            TAcc,
            TBias,
            TOut,
            decltype(conv_desc),
            decltype(conv_tuning_parameters)>{};

    for(int i = 0; i < 5; i++)
    {

        const auto ave_time =
            conv_driver.Run(static_cast<TInWei*>(wei_k_c0_y_x_c1_device_buf.GetDeviceBuffer()),
                            static_cast<TInWei*>(in_n_c0_hi_wi_c1_device_buf.GetDeviceBuffer()),
                            static_cast<TBias*>(bias_k0_k1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(add_n_k0_hox2_wox2_k1_device_buf.GetDeviceBuffer()),
                            nrepeat);

        {
            float perf = static_cast<float>(std::size_t(2) * N * K * Ho * Wo * C0 * C1 * Y * X) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }

    add_n_k0_hox2_wox2_k1_device_buf.FromDevice(add_n_k0_hox2_wox2_k1_out.mData.data());
}
