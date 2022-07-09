#ifndef DRIVER_CONVOLUTION_MAXPOOL_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NC0HWc1_KC0YXC1_NK0HWK1_HPP
#define DRIVER_CONVOLUTION_MAXPOOL_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NC0HWc1_KC0YXC1_NK0HWK1_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_dlops_v3.hpp"

namespace ck {

template <typename InLengths,
          typename WeiLengths,
          typename OutLengths,
          typename MaxLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
struct ConvBiasActivMaxPoolDesc
{
#if 0
    InLengths in_n_c0_hi_wi_c1_desc;
    WeiLengths wei_c0_y_x_k_c1_desc;
    OutLengths out_n_k0_ho_wo_k1_desc;

    ConvStrides conv_strides;
    ConvDilations conv_dilations;

    InLeftPads in_left_pads;
    InRightPads in_right_pads;


    ConvBiasActivDesc(InLengths in_n_c0_hi_wi_c1_desc_,
                      WeiLengths wei_c0_y_x_k_c1_desc_,
                      OutLengths out_n_k0_ho_wo_k1_desc_,
                      ConvStrides conv_strides_,
                      ConvDilations conv_dilations_,
                      InLeftPads in_left_pads_,
                      InRightPads in_right_pads_)
    {
        in_n_c0_hi_wi_c1_desc  = in_n_c0_hi_wi_c1_desc_;
        wei_c0_y_x_k_c1_desc   = wei_c0_y_x_k_c1_desc_;
        out_n_k0_ho_wo_k1_desc = out_n_k0_ho_wo_k1_desc_;
        conv_strides           = conv_strides_;
        conv_dilations         = conv_dilations_;
        in_left_pads           = in_left_pads_;
        in_right_pads          = in_right_pads_;
    }
#else
    static constexpr auto in_n_c0_hi_wi_c1_desc  = InLengths{};
    static constexpr auto wei_c0_y_x_k_c1_desc   = WeiLengths{};
    static constexpr auto out_n_k0_ho_wo_k1_desc = OutLengths{};
    static constexpr auto max_n_k0_hx_wx_k1_desc = MaxLengths{};

    static constexpr auto conv_strides   = ConvStrides{};
    static constexpr auto conv_dilations = ConvDilations{};

    static constexpr auto in_left_pads  = InLeftPads{};
    static constexpr auto in_right_pads = InRightPads{};
#endif

    void printConvDesc()
    {
        using namespace ck;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        const auto N  = in_n_c0_hi_wi_c1_desc.GetLength(I0);
        const auto C0 = in_n_c0_hi_wi_c1_desc.GetLength(I1);
        const auto Hi = in_n_c0_hi_wi_c1_desc.GetLength(I2);
        const auto Wi = in_n_c0_hi_wi_c1_desc.GetLength(I3);
        const auto C1 = in_n_c0_hi_wi_c1_desc.GetLength(I4);

        const auto K0 = out_n_k0_ho_wo_k1_desc.GetLength(I1);
        const auto Ho = out_n_k0_ho_wo_k1_desc.GetLength(I2);
        const auto Wo = out_n_k0_ho_wo_k1_desc.GetLength(I3);
        const auto K1 = out_n_k0_ho_wo_k1_desc.GetLength(I4);

        const auto K = wei_c0_y_x_k_c1_desc.GetLength(I0);
        const auto Y = wei_c0_y_x_k_c1_desc.GetLength(I2);
        const auto X = wei_c0_y_x_k_c1_desc.GetLength(I3);

        const auto ConvStrideH = conv_strides[I0];
        const auto ConvStrideW = conv_strides[I1];

        const auto ConvDilationH = conv_dilations[I0];
        const auto ConvDilationW = conv_dilations[I1];

        std::cout << "input_"
                  << "n" << N << "c" << C0 << "h" << Hi << "w" << Wi << "c" << C1 << "_filter_k"
                  << K << "c" << C0 << "y" << Y << "x" << X << "c" << C1 << "_out_n" << N << "k"
                  << K0 << "h" << Ho << "w" << Wo << "k" << K1 << std::endl;

        std::cout << "ConvStride = " << ConvStrideH << "," << ConvStrideW << std::endl;
        std::cout << "ConvDilation = " << ConvDilationH << "," << ConvDilationW << std::endl;
    }
};

template <typename GridwiseGemmDesc,
          typename AE0E1K0K1E2GridDesc,
          typename BE0E1NH0H1H2W0W1W2E2GridDesc,
          typename BiasK0K1GridDesc,
          typename CK0K1NH0H1H2W0W1W2GridDesc,
          typename DK0K1NH0H1HxW0W1WxGridDesc,
          typename CBlockIdToKNHoWoBlockClusterAdaptor,
          index_t GridSize,
          index_t BlockSize>
struct GemmBiasActivMaxPoolArguments
{
    static constexpr auto gridwise_gemm_desc = GridwiseGemmDesc{};

    // AE0E1K0K1E2GridDesc a_e0_e1_k0_k1_e2_grid_desc;
    // BE0E1NH0H1H2W0W1W2E2GridDesc b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc;
    // CK0K1NH0H1H2W0W1W2GridDesc c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc;
    // CBlockIdToKNHoWoBlockClusterAdaptor c_blockid_to_k_n_h_w_block_cluster_adaptor;

    static constexpr auto a_e0_e1_k0_k1_e2_grid_desc               = AE0E1K0K1E2GridDesc{};
    static constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc = BE0E1NH0H1H2W0W1W2E2GridDesc{};
    static constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc    = CK0K1NH0H1H2W0W1W2GridDesc{};
    static constexpr auto bias_k0_k1_grid_desc                     = BiasK0K1GridDesc{};
    static constexpr auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc    = DK0K1NH0H1HxW0W1WxGridDesc{};

    static constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToKNHoWoBlockClusterAdaptor{};

    static constexpr index_t grid_size  = GridSize;
    static constexpr index_t block_size = BlockSize;
    // bool has_main_e0_block_loop;

#if 0
    GemmBiasActivArguments(
        // const AE0E1K0K1E2GridDesc a_e0_e1_k0_k1_e2_grid_desc_,
        // const BE0E1NH0H1H2W0W1W2E2GridDesc b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc_,
        // const CK0K1NH0H1H2W0W1W2GridDesc c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc_,
        // const CBlockIdToKNHoWoBlockClusterAdaptor c_blockid_to_k_n_h_w_block_cluster_adaptor_,
        const index_t grid_size_,
        const index_t block_size_)
    {
        // a_e0_e1_k0_k1_e2_grid_desc                 = a_e0_e1_k0_k1_e2_grid_desc_;
        // b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc   = b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc_;
        // c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc      = c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc_;
        // c_blockid_to_k_n_h_w_block_cluster_adaptor = c_blockid_to_k_n_h_w_block_cluster_adaptor_;

        grid_size  = grid_size_;
        block_size = block_size_;
        // has_main_e0_block_loop = has_main_e0_block_loop_;
    }
#endif
};

template <typename CThreadBuff,
          typename DGlobalBuff,
          typename CBlockIndex,
          typename CThreadIndex,
          typename CThreadDesc_K1_N_H2_W2,
          typename DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx>
__device__ static void
MaxPool2x2(const CThreadBuff& c_thread_buf,
           DGlobalBuff& d_global_buf,
           const CBlockIndex& c_block_idx,
           const CThreadIndex& c_thread_idx,
           const CThreadDesc_K1_N_H2_W2&,
           const DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx& d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc)
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    const index_t k_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I0]);
    const index_t n_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I1]);
    const index_t ho_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I2]);
    const index_t wo_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I3]);

    const auto k_thread_id  = c_thread_idx[I0];
    const auto ho_thread_id = c_thread_idx[I2];
    const auto wo_thread_id = c_thread_idx[I3];

    constexpr auto c_k1_n_h2_w2_thread_gemm_desc = CThreadDesc_K1_N_H2_W2{};

    constexpr auto KPerThread  = CThreadDesc_K1_N_H2_W2{}.GetLength(I0);
    constexpr auto HoPerThread = CThreadDesc_K1_N_H2_W2{}.GetLength(I2);
    constexpr auto WoPerThread = CThreadDesc_K1_N_H2_W2{}.GetLength(I3);

    static_assert(HoPerThread % 2 == 0 && WoPerThread % 2 == 0, "");

    using FloatC = remove_cvref_t<typename DGlobalBuff::type>;

    constexpr auto HoPerThread_2 = HoPerThread / 2;
    constexpr auto WoPerThread_2 = WoPerThread / 2;

    constexpr auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc =
        make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                       Number<KPerThread>{},
                                                       I1,
                                                       I1,
                                                       I1,
                                                       Number<HoPerThread_2>{},
                                                       I1,
                                                       I1,
                                                       Number<WoPerThread_2>{}));

    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatC,
                 d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc.GetElementSpaceSize(),
                 true>
        d_thread_buf;

    static_for<0, KPerThread, 1>{}([&](auto ki) {
        static_for<0, HoPerThread_2, 1>{}([&](auto hi) {
            static_for<0, WoPerThread_2, 1>{}([&](auto wi) {
                constexpr index_t d_offset =
                    d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc.CalculateOffset(
                        make_tuple(0, ki, 0, 0, 0, hi, 0, 0, wi));

                constexpr index_t c_offset_0 = c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(
                    make_tuple(ki, 0, hi * 2, wi * 2));
                constexpr index_t c_offset_1 = c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(
                    make_tuple(ki, 0, hi * 2, wi * 2 + 1));
                constexpr index_t c_offset_2 = c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(
                    make_tuple(ki, 0, hi * 2 + 1, wi * 2));
                constexpr index_t c_offset_3 = c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(
                    make_tuple(ki, 0, hi * 2 + 1, wi * 2 + 1));

                auto v0 = c_thread_buf[Number<c_offset_0>{}];
                auto v1 = c_thread_buf[Number<c_offset_1>{}];
                auto v2 = c_thread_buf[Number<c_offset_2>{}];
                auto v3 = c_thread_buf[Number<c_offset_3>{}];

                d_thread_buf(Number<d_offset>{}) = max_(v0, max_(v1, max_(v2, v3)));
            });
        });
    });

    const index_t k_thread_data_on_global = k_thread_id * KPerThread;

    ThreadwiseTensorSliceTransfer_v1r3<
        FloatC,
        FloatC,
        decltype(d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc),
        decltype(d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc),
        ck::tensor_operation::element_wise::PassThrough,
        Sequence<I1, KPerThread, I1, I1, I1, HoPerThread_2, I1, I1, WoPerThread_2>,
        Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>, // CThreadTransferSrcDstAccessOrder
        1,                                   // CThreadTransferSrcDstVectorDim
        8,                                   // CThreadTransferDstScalarPerVector_K
        InMemoryDataOperationEnum_t::Set,
        1,
        true>(d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc,
              make_multi_index(k_block_work_id,
                               k_thread_data_on_global,
                               n_block_work_id,
                               ho_block_work_id,
                               ho_thread_id,
                               0,
                               wo_block_work_id,
                               wo_thread_id,
                               0),
              ck::tensor_operation::element_wise::PassThrough{})
        .Run(d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc,
             make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
             d_thread_buf,
             d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc,
             d_global_buf);
}

template <typename GemmBiasActivMaxPoolArguments,
          typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatC>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_conv_bias_activ_maxpool_dlops_v3(const FloatAB* __restrict__ p_a_grid,
                                                const FloatAB* __restrict__ p_b_grid,
                                                const FloatBias* __restrict__ p_bias_grid,
                                                FloatC* __restrict__ p_c_grid,
                                                FloatC* __restrict__ p_d_grid,
                                                float scaleGemm)
{

    constexpr auto gemm_bias_activ_maxpool_arg = GemmBiasActivMaxPoolArguments{};

    using GridwiseGemm = decltype(gemm_bias_activ_maxpool_arg.gridwise_gemm_desc);

    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc =
        gemm_bias_activ_maxpool_arg.a_e0_e1_k0_k1_e2_grid_desc;
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        gemm_bias_activ_maxpool_arg.b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc;
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc =
        gemm_bias_activ_maxpool_arg.c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc;
    constexpr auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc =
        gemm_bias_activ_maxpool_arg.d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc;

    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        gemm_bias_activ_maxpool_arg.c_blockid_to_k_n_h_w_block_cluster_adaptor;

    const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_a_grid, a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize());
    const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_b_grid, b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize());

    constexpr auto c_k1_n_h2_w2_thread_gemm_desc = GridwiseGemm::MakeCK1NH2W2ThreadDescriptor();

    // register allocation for output
    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatAcc,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        c_thread_buf;

    const index_t block_id = get_block_1d_id();

    const auto c_k_n_h_w_block_cluster_idx =
        GridwiseGemm::GetCBlockIndex(c_blockid_to_k_n_h_w_block_cluster_adaptor, block_id);

    const auto c_thread_mtx_index = GridwiseGemm::GetCThreadIndex();

    // GemmOp
    GridwiseGemm::GemmOpHasE1Loop(a_global_buf,
                                  b_global_buf,
                                  c_thread_buf,
                                  p_shared_block,
                                  c_k_n_h_w_block_cluster_idx,
                                  c_thread_mtx_index,
                                  a_e0_e1_k0_k1_e2_grid_desc,
                                  b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                  c_k1_n_h2_w2_thread_gemm_desc);

    const auto bias_k0_k1_grid_desc =
        GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_bias_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    // Bias
    GridwiseGemm::BiasOp(bias_global_buf,
                         c_thread_buf,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         bias_k0_k1_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc);
    // Activ
    GridwiseGemm::ActivOp(c_thread_buf,
                          c_k1_n_h2_w2_thread_gemm_desc,
                          ck::tensor_operation::element_wise::RequantHardTanh{scaleGemm});

    auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_c_grid, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    // Output
    GridwiseGemm::WriteOut(c_thread_buf,
                           c_global_buf,
                           c_k_n_h_w_block_cluster_idx,
                           c_thread_mtx_index,
                           c_k1_n_h2_w2_thread_gemm_desc,
                           c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    auto d_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_d_grid, d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc.GetElementSpaceSize());

    // MaxPool
    MaxPool2x2(c_thread_buf,
               d_global_buf,
               c_k_n_h_w_block_cluster_idx,
               c_thread_mtx_index,
               c_k1_n_h2_w2_thread_gemm_desc,
               d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc);
}

template <typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatC,
          typename ConvBiasActivMaxPoolDesc,
          typename GridGemmTuningParameters>
struct DriverDynamicConvolutionForwardImplicitGemmDlops_v5r1_nc0hwc1_kc0yxc1_nk0hwk1_maxpool
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    static constexpr auto BlockSize = Number<GridGemmTuningParameters::BlockSize>{};

    static constexpr auto E1 = Number<GridGemmTuningParameters::E1>{};
    static constexpr auto E2 = Number<GridGemmTuningParameters::E2>{};
    static constexpr auto K2 = Number<GridGemmTuningParameters::K2>{};

    static constexpr auto E0PerBlock = Number<GridGemmTuningParameters::E0PerBlock>{};
    static constexpr auto KPerBlock  = Number<GridGemmTuningParameters::KPerBlock>{};
    static constexpr auto HoPerBlock = Number<GridGemmTuningParameters::HoPerBlock>{};
    static constexpr auto WoPerBlock = Number<GridGemmTuningParameters::WoPerBlock>{};
    static constexpr auto E1PerBlock = Number<GridGemmTuningParameters::E1PerBlock>{};

    static constexpr auto KPerThread  = Number<GridGemmTuningParameters::KPerThread>{};
    static constexpr auto HoPerThread = Number<GridGemmTuningParameters::HoPerThread>{};
    static constexpr auto WoPerThread = Number<GridGemmTuningParameters::WoPerThread>{};
    static constexpr auto EPerThread  = Number<GridGemmTuningParameters::EPerThread>{};

    using ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2 =
        decltype(GridGemmTuningParameters::ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2);
    using ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 =
        decltype(GridGemmTuningParameters::ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2);

    static constexpr auto ABlockTransferSrcScalarPerVector_E2 =
        GridGemmTuningParameters::ABlockTransferSrcScalarPerVector_E2;
    static constexpr auto ABlockTransferDstScalarPerVector_E2 =
        GridGemmTuningParameters::ABlockTransferDstScalarPerVector_E2;
    static constexpr auto BThreadTransferSrcScalarPerVector_E2 =
        GridGemmTuningParameters::BThreadTransferSrcScalarPerVector_E2;
    static constexpr auto CThreadTransferDstScalarPerVector_K =
        GridGemmTuningParameters::CThreadTransferDstScalarPerVector_K;

    template <typename DGridDesc_K_N_Hx_Wx>
    __host__ __device__ static constexpr auto
    MakeDK0K1NH0H1HxW0W1WxGridDescriptorMaxPool(const DGridDesc_K_N_Hx_Wx& d_k_n_hx_wx_grid_desc)
    {
        const auto K  = d_k_n_hx_wx_grid_desc.GetLength(I0);
        const auto N  = d_k_n_hx_wx_grid_desc.GetLength(I1);
        const auto Hx = d_k_n_hx_wx_grid_desc.GetLength(I2);
        const auto Wx = d_k_n_hx_wx_grid_desc.GetLength(I3);

        const auto K1 = Number<KPerBlock>{};
        const auto K0 = K / K1;

        const auto H2 = Number<HoPerThread / 2>{};
        const auto H1 = Number<HoPerBlock / HoPerThread>{};
        const auto H0 = Number<Hx / (H1 * H2)>{};

        const auto W2 = Number<WoPerThread / 2>{};
        const auto W1 = Number<WoPerBlock / WoPerThread>{};
        const auto W0 = Number<Wx / (W1 * W2)>{};

        const auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc = transform_tensor_descriptor(
            d_k_n_hx_wx_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_unmerge_transform(make_tuple(H0, H1, H2)),
                       make_unmerge_transform(make_tuple(W0, W1, W2))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3, 4, 5>{}, Sequence<6, 7, 8>{}));

        return d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc;
    }

    constexpr auto MakeGridwiseGemmBiasActivMaxPool() const
    {
        constexpr auto conv_desc = ConvBiasActivMaxPoolDesc{};

        constexpr auto in_n_c0_hi_wi_c1_global_desc = conv_desc.in_n_c0_hi_wi_c1_desc;

        constexpr auto N  = Number<in_n_c0_hi_wi_c1_global_desc.GetLength(I0)>{};
        constexpr auto C0 = Number<in_n_c0_hi_wi_c1_global_desc.GetLength(I1)>{};
        constexpr auto Hi = Number<in_n_c0_hi_wi_c1_global_desc.GetLength(I2)>{};
        constexpr auto Wi = Number<in_n_c0_hi_wi_c1_global_desc.GetLength(I3)>{};
        // constexpr auto C1 = Number<in_n_c0_hi_wi_c1_global_desc.GetLength(I4)>{};

        constexpr auto out_n_k0_ho_wo_k1_global_desc = conv_desc.out_n_k0_ho_wo_k1_desc;

        constexpr auto K0 = Number<out_n_k0_ho_wo_k1_global_desc.GetLength(I1)>{};
        constexpr auto Ho = Number<out_n_k0_ho_wo_k1_global_desc.GetLength(I2)>{};
        constexpr auto Wo = Number<out_n_k0_ho_wo_k1_global_desc.GetLength(I3)>{};
        constexpr auto K1 = Number<out_n_k0_ho_wo_k1_global_desc.GetLength(I4)>{};

        constexpr auto wei_k_c0_y_x_c1_global_desc = conv_desc.wei_c0_y_x_k_c1_desc;

        constexpr auto K = Number<wei_k_c0_y_x_c1_global_desc.GetLength(I0)>{};
        constexpr auto Y = Number<wei_k_c0_y_x_c1_global_desc.GetLength(I2)>{};
        constexpr auto X = Number<wei_k_c0_y_x_c1_global_desc.GetLength(I3)>{};

        constexpr auto max_n_k0_hx_wx_k1_global_desc = conv_desc.max_n_k0_hx_wx_k1_desc;

        constexpr auto Hx = max_n_k0_hx_wx_k1_global_desc.GetLength(I2);
        constexpr auto Wx = max_n_k0_hx_wx_k1_global_desc.GetLength(I3);

        constexpr auto ConvStrideH = Number<conv_desc.conv_strides[I0]>{};
        constexpr auto ConvStrideW = Number<conv_desc.conv_strides[I1]>{};

        constexpr auto ConvDilationH = Number<conv_desc.conv_dilations[I0]>{};
        constexpr auto ConvDilationW = Number<conv_desc.conv_dilations[I1]>{};

        constexpr auto Hop = Number<(Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock>{};
        constexpr auto Wop = Number<(Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock>{};

        constexpr auto InLeftPadH = Number<conv_desc.in_left_pads[I0]>{};
        constexpr auto InLeftPadW = Number<conv_desc.in_left_pads[I1]>{};

        constexpr auto OutRightPadH = Hop - Ho;
        constexpr auto OutRightPadW = Wop - Wo;

        const auto OutRightPadHx = Number<OutRightPadH / 2>{};
        const auto OutRightPadWx = Number<OutRightPadW / 2>{};

        constexpr auto InRightPadH =
            Number<conv_desc.in_right_pads[I0] + OutRightPadH * ConvStrideH>{};
        constexpr auto InRightPadW =
            Number<conv_desc.in_right_pads[I1] + OutRightPadW * ConvStrideW>{};

        if((C0 * Y * X) % (E1 * E0PerBlock) != 0)
        {
            std::cerr << "E = " << C0 * Y * X << " E1 = " << E1 << " E0PerBlock = " << E0PerBlock
                      << std::endl;
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        constexpr auto E  = Number<C0 * Y * X>{};
        constexpr auto E0 = Number<E / E1>{};

        // weight tensor
        constexpr auto a_e_k_e2_grid_desc =
            transform_tensor_descriptor(make_naive_tensor_descriptor_packed(make_tuple(K, E, E2)),
                                        make_tuple(make_pass_through_transform(K),
                                                   make_pass_through_transform(E),
                                                   make_pass_through_transform(E2)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<1>{}, Sequence<0>{}, Sequence<2>{}));

        static_assert(a_e_k_e2_grid_desc.IsKnownAtCompileTime(), "");

        constexpr auto a_e0_e1_k_e2_grid_desc =
            transform_tensor_descriptor(a_e_k_e2_grid_desc,
                                        make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                                                   make_pass_through_transform(K),
                                                   make_pass_through_transform(E2)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));

        static_assert(a_e0_e1_k_e2_grid_desc.IsKnownAtCompileTime(), "");

        // input tensor
        constexpr auto in_n_c0_hip_wip_e2_global_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(N, C0, Hi, Wi, E2)),
            make_tuple(make_pass_through_transform(N),
                       make_pass_through_transform(C0),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        static_assert(in_n_c0_hip_wip_e2_global_desc.IsKnownAtCompileTime(), "");

        constexpr auto in_n_c0_y_ho_x_wo_e2_global_desc = transform_tensor_descriptor(
            in_n_c0_hip_wip_e2_global_desc,
            make_tuple(
                make_pass_through_transform(N),
                make_pass_through_transform(C0),
                make_embed_transform(make_tuple(Y, Hop), make_tuple(ConvDilationH, ConvStrideH)),
                make_embed_transform(make_tuple(X, Wop), make_tuple(ConvDilationW, ConvStrideW)),
                make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}, Sequence<6>{}));

        static_assert(in_n_c0_y_ho_x_wo_e2_global_desc.IsKnownAtCompileTime(), "");

        constexpr auto in_e_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
            in_n_c0_y_ho_x_wo_e2_global_desc,
            make_tuple(make_merge_transform(make_tuple(C0, Y, X)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Hop),
                       make_pass_through_transform(Wop),
                       make_pass_through_transform(E2)),
            make_tuple(
                Sequence<1, 2, 4>{}, Sequence<0>{}, Sequence<3>{}, Sequence<5>{}, Sequence<6>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        static_assert(in_e_n_ho_wo_e2_grid_desc.IsKnownAtCompileTime(), "");

        constexpr auto b_e0_e1_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
            in_e_n_ho_wo_e2_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Hop),
                       make_pass_through_transform(Wop),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}));

        static_assert(b_e0_e1_n_ho_wo_e2_grid_desc.IsKnownAtCompileTime(), "");

        // output tensor
        constexpr auto c_k_n_hop_wop_grid_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1)),
            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_pad_transform(Ho, I0, OutRightPadH),
                       make_pad_transform(Wo, I0, OutRightPadW)),
            make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        // max tensor
        const auto d_k_n_hx_wx_grid_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(N, K0, Hx, Wx, K1)),
            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_pad_transform(Hx, I0, OutRightPadHx),
                       make_pad_transform(Wx, I0, OutRightPadWx)),
            make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        std::cerr << "Hop = " << Hop << " Wop = " << Wop << std::endl;

        if(!((K % KPerBlock) == 0 && (Hop % HoPerBlock) == 0 && (Wop % WoPerBlock) == 0 &&
             (E1 % E1PerBlock) == 0))
        {
            std::cerr << "K = " << K << " KPerBlock = " << KPerBlock << " Hop = " << Hop
                      << " HoPerBlock = " << HoPerBlock << " Wop = " << Wop
                      << " WoPerBlock = " << WoPerBlock << " E1 = " << E1
                      << " E1PerBlock = " << E1PerBlock << std::endl;
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        std::cerr << "a_size = " << a_e0_e1_k_e2_grid_desc.GetElementSpaceSize() * sizeof(FloatAB)
                  << ", b_size = "
                  << b_e0_e1_n_ho_wo_e2_grid_desc.GetElementSpaceSize() * sizeof(FloatAB)
                  << ", c = " << c_k_n_hop_wop_grid_desc.GetElementSpaceSize() * sizeof(FloatC)
                  << std::endl;

        // GEMM
        using GridwiseGemm = GridwiseGemmDlops_km_kn_mn_v3<
            BlockSize,
            FloatAB,
            FloatAcc,
            FloatC,
            InMemoryDataOperationEnum_t::Set,
            decltype(a_e0_e1_k_e2_grid_desc),
            decltype(b_e0_e1_n_ho_wo_e2_grid_desc),
            decltype(c_k_n_hop_wop_grid_desc),
            E1,
            E2,
            K2,
            KPerBlock,
            HoPerBlock,
            WoPerBlock,
            E0PerBlock,
            E1PerBlock,
            KPerThread,
            HoPerThread,
            WoPerThread,
            EPerThread,
            ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2,
            ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
            Sequence<2, 3, 0, 1, 4>,
            Sequence<0, 1, 2, 3, 4>,
            4,
            ABlockTransferSrcScalarPerVector_E2,
            ABlockTransferDstScalarPerVector_E2,
            false, // don't move back src coordinate after threadwise copy
            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>, // E0, E1, N, H0, H1, H2, W0, W1, W2, E2
            9,
            BThreadTransferSrcScalarPerVector_E2,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
            // MoveSrcSliceWindow() to save addr computation
            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>, // K0, K1, N, H0, H1, H2, W0, W1, W2
            1,
            CThreadTransferDstScalarPerVector_K>;

        static_assert(c_k_n_hop_wop_grid_desc.IsKnownAtCompileTime(), "");

        const auto a_e0_e1_k0_k1_e2_grid_desc =
            GridwiseGemm::MakeAE0E1K0K1E2GridDescriptor(a_e0_e1_k_e2_grid_desc);
        const auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
            GridwiseGemm::MakeBE0E1NH0H1H2W0W1W2E2GridDescriptor(b_e0_e1_n_ho_wo_e2_grid_desc);
        const auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc =
            GridwiseGemm::MakeCK0K1NH0H1H2W0W1W2GridDescriptor(c_k_n_hop_wop_grid_desc);
        const auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc =
            MakeDK0K1NH0H1HxW0W1WxGridDescriptorMaxPool(d_k_n_hx_wx_grid_desc);

        static_assert(c_k_n_hop_wop_grid_desc.IsKnownAtCompileTime(), "");

        const auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
            GridwiseGemm::MakeCBlockIdToKNHoWoBlockClusterAdaptor(c_k_n_hop_wop_grid_desc);

        const auto bias_k0_k1_grid_desc =
            GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

        const auto grid_size = (K / KPerBlock) * (Hop / HoPerBlock) * (Wop / WoPerBlock) * N;

        const bool has_main_e0_block_loop = E0 > 1;

        std::cerr << "grid_size = " << grid_size
                  << " has_main_e0_block_loop = " << has_main_e0_block_loop << std::endl;

        static_assert(a_e0_e1_k0_k1_e2_grid_desc.IsKnownAtCompileTime(), "");
        static_assert(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.IsKnownAtCompileTime(), "");
        static_assert(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.IsKnownAtCompileTime(), "");
        static_assert(bias_k0_k1_grid_desc.IsKnownAtCompileTime(), "");
        static_assert(c_blockid_to_k_n_h_w_block_cluster_adaptor.IsKnownAtCompileTime(), "");

        using AGridDesc_E0_E1_K0_K1_E2 = decltype(a_e0_e1_k0_k1_e2_grid_desc);
        using BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2 =
            decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc);
        using CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2 = decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);
        using BiasDesc_K0_K1                      = decltype(bias_k0_k1_grid_desc);
        using CBlockIdToBlockClusterAdaptor_K_N_H_W =
            decltype(c_blockid_to_k_n_h_w_block_cluster_adaptor);
        using DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx = decltype(d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc);

        return GemmBiasActivMaxPoolArguments<GridwiseGemm,
                                             AGridDesc_E0_E1_K0_K1_E2,
                                             BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
                                             BiasDesc_K0_K1,
                                             CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
                                             DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx,
                                             CBlockIdToBlockClusterAdaptor_K_N_H_W,
                                             grid_size,
                                             BlockSize>{};
    }

    __host__ float Run(const FloatAB* __restrict__ p_a_grid,
                       const FloatAB* __restrict__ p_b_grid,
                       const FloatBias* __restrict__ p_bias_grid,
                       FloatC* __restrict__ p_c_grid,
                       FloatC* __restrict__ p_d_grid,
                       const int nrepeat) const
    {

        const auto gemm_bias_activ_maxpool_args = MakeGridwiseGemmBiasActivMaxPool();

        const auto kernel =
            kernel_conv_bias_activ_maxpool_dlops_v3<decltype(gemm_bias_activ_maxpool_args),
                                                    FloatAB,
                                                    FloatAcc,
                                                    FloatBias,
                                                    FloatC>;

        auto ave_time = launch_and_time_kernel(kernel,
                                               nrepeat,
                                               dim3(gemm_bias_activ_maxpool_args.grid_size),
                                               dim3(gemm_bias_activ_maxpool_args.block_size),
                                               0,
                                               p_a_grid,
                                               p_b_grid,
                                               p_bias_grid,
                                               p_c_grid,
                                               p_d_grid,
                                               0.3);
        return ave_time;
    }
};

} // namespace ck
#endif
