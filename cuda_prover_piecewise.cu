#include <string>
#include <chrono>

#define NDEBUG 1

#include <prover_reference_functions.hpp>
#include "multiexp/reduce.cu"
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
// #include <libff/algebra/curves/mnt753/mnt4753/mnt4753_init.hpp>
// #include <libff/algebra/curves/mnt753/mnt6753/mnt6753_init.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>
#include <libfqfft/tools/exceptions.hpp>
#include <libfqfft/evaluation_domain/evaluation_domain.hpp>

#include "fft.cu"
// This is where all the FFTs happen

static inline auto now() -> decltype(std::chrono::high_resolution_clock::now()) {
    return std::chrono::high_resolution_clock::now();
}

template<typename T>
void
print_time(T &t1, const char *str) {
    auto t2 = std::chrono::high_resolution_clock::now();
    auto tim = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("%s: %ld ms\n", str, tim);
    t1 = t2;
}

template <typename B>
typename B::vector_Fr *compute_H(size_t d, typename B::vector_Fr *ca,
                                 typename B::vector_Fr *cb,
                                 typename B::vector_Fr *cc) {
  auto domain = B::get_evaluation_domain(d + 1);

  B::domain_iFFT(domain, ca);
  B::domain_iFFT(domain, cb);

  B::domain_cosetFFT(domain, ca);
  B::domain_cosetFFT(domain, cb);

  //cudaStreamSynchronize(A);
  // Use ca to store H
  auto H_tmp = ca;

  size_t m = B::domain_get_m(domain);
  // for i in 0 to m: H_tmp[i] *= cb[i]
  B::vector_Fr_muleq(H_tmp, cb, m);

  B::domain_iFFT(domain, cc);
  B::domain_cosetFFT(domain, cc);

  m = B::domain_get_m(domain);

  // for i in 0 to m: H_tmp[i] -= cc[i]
  B::vector_Fr_subeq(H_tmp, cc, m);

  B::domain_divide_by_Z_on_coset(domain, H_tmp);

  B::domain_icosetFFT(domain, H_tmp);

  m = B::domain_get_m(domain);
  typename B::vector_Fr *H_res = B::vector_Fr_zeros(m + 1);
  B::vector_Fr_copy_into(H_tmp, H_res, m);
  return H_res;
}

// TODO: change these to use gpu Field interfaces correctly
template <typename FieldT>
__global__ void vector_Fr_muleq(var *x, var *y, size_t n)
{
    // int i = blockIdx.x*blockDim.x + threadIdx.x;
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;
    if (idx < n) {
        FieldT q;
        FieldT r;

        int off = idx * ELT_LIMBS;
        FieldT::load(q, x + off);
        FieldT::load(r, y + off);
        
        // TODO: check if needed
        FieldT::from_monty(r, r);
        FieldT::mul(q, q, r); 

        FieldT::store(x + off, q);
    }
}

template <typename FieldT>
__global__ void vector_Fr_subeq(var *x, var *y, size_t n)
{
    // int i = blockIdx.x*blockDim.x + threadIdx.x;
    // if (i < n) {
    //     FieldT::sub(x + i, x + i, y + i); 
    // }
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;
    if (idx < n) {
        FieldT q;
        FieldT r;

        int off = idx * ELT_LIMBS;
        FieldT::load(q, x + off);
        FieldT::load(r, y + off);

        // TODO: check if needed
        FieldT::from_monty(r, r);
        FieldT::sub(q, q, r); 

        FieldT::store(x + off, q);
    }
}

template <typename B, typename EC>
var *compute_H_gpu(size_t d, var *gpu_ca,var *gpu_cb, var *gpu_cc,
                                 const var *gen,
                                 const var *gen_inv, 
                                 const var *omega,
                                 const var *omega_inv,
                                 const var *sconst,
                                 const var *z_inverse) {

    typedef typename EC::field_type FF;
  
    auto domain = B::get_evaluation_domain(d + 1);
    auto beginning = now();
    auto t = beginning;
//   B::domain_iFFT(domain, ca);
//   B::domain_iFFT(domain, cb);

    iFFT<FF>(gpu_ca, omega_inv, sconst, d+1);
    iFFT<FF>(gpu_cb, omega_inv, sconst, d+1);
    print_time(t, "iFFT ca and cosetFFT cb");

    domain_cosetFFT_gpu<B, FF>(gpu_ca, d+1, gen, omega);
    print_time(t, "gpu cosetFFT ca");
  //domain_cosetFFT_gpu<B, typename B::vector_Fr, libff::Fr<libff::mnt4753_pp>>(ca, d);
//   B::domain_cosetFFT_gpu(domain, ca, d + 1);
  //B::domain_cosetFFT(domain, ca);
//   B::domain_cosetFFT(domain, cb);
    domain_cosetFFT_gpu<B, FF>(gpu_cb, d+1, gen, omega);
    print_time(t, "gpu cosetFFT cb");

  // Use ca to store H
//   auto H_tmp = ca;
    var *H_tmp = gpu_ca;
    printf("Address of H_tmp is %p\n", (void *)H_tmp); 
//   size_t m = B::domain_get_m(domain);
  // for i in 0 to m: H_tmp[i] *= cb[i]
//   B::vector_Fr_muleq(H_tmp, cb, m);

    // TODO: commmenting out for testing
    vector_Fr_muleq<FF><<<2, 256>>>(H_tmp, gpu_cb, d+1);

    cudaDeviceSynchronize();
    print_time(t, "gpu vector Fr muleq H_tmp and cb");

    // var *cb = (var *)malloc((d+1)*ELT_BYTES);
    // cudaMemcpy((void **)&cb[0], gpu_cb, (d+1)*ELT_BYTES, cudaMemcpyDeviceToHost);
    // printf("value gpu_cb: %d ", *cb);
//   B::domain_iFFT(domain, cc);
//   B::domain_cosetFFT(domain, cc);
    iFFT<FF>(gpu_cc, omega_inv, sconst, d+1);
    domain_cosetFFT_gpu<B, FF>(gpu_cc, d+1, gen, omega);
    print_time(t, "iFFT cc and cosetFFT cc");

//   m = B::domain_get_m(domain);

  // for i in 0 to m: H_tmp[i] -= cc[i]
//   B::vector_Fr_subeq(H_tmp, cc, m);
    vector_Fr_subeq<FF><<<2, 256>>>(H_tmp, gpu_cc, d+1);
    cudaDeviceSynchronize();
    print_time(t, "gpu vector Fr subeq H_tmp and cc");
//   B::domain_divide_by_Z_on_coset(domain, H_tmp);
    // TODO: change currently doing nothing
    gpu_divide_by_Z_on_coset<FF><<<2, 256>>>(H_tmp, z_inverse, d+1);
    cudaDeviceSynchronize();
    print_time(t, "gpu divide by z on coset H_tmp");
//   B::domain_icosetFFT(domain, H_tmp);
    icosetFFT<FF>(H_tmp, omega_inv, sconst, gen_inv, d+1);
    print_time(t, "icosetFFT H_tmp");
//   m = B::domain_get_m(domain);
//   typename B::vector_Fr *H_res = B::vector_Fr_zeros(m + 1);
//   B::vector_Fr_copy_into(H_tmp, H_res, m);
    return H_tmp;
}

static size_t read_size_t(FILE* input) {
  size_t n;
  fread((void *) &n, sizeof(size_t), 1, input);
  return n;
}

template< typename B >
struct ec_type;

template<>
struct ec_type<mnt4753_libsnark> {
    typedef ECp_MNT4 ECp;
    typedef ECp2_MNT4 ECpe;
};

template<>
struct ec_type<mnt6753_libsnark> {
    typedef ECp_MNT6 ECp;
    typedef ECp3_MNT6 ECpe;
};


void
check_trailing(FILE *f, const char *name) {
    long bytes_remaining = 0;
    while (fgetc(f) != EOF)
        ++bytes_remaining;
    if (bytes_remaining > 0)
        fprintf(stderr, "!! Trailing characters in \"%s\": %ld\n", name, bytes_remaining);
}

template <typename B>
void run_prover_gpu(
        const char *params_path,
        const char *input_path,
        const char *output_path,
        const char *preprocessed_path)
{
    B::init_public_params();

    size_t primary_input_size = 1;

    auto beginning = now();
    auto t = beginning;

    FILE *params_file = fopen(params_path, "r");
    size_t d = read_size_t(params_file);
    size_t m = read_size_t(params_file);
    rewind(params_file);

    printf("d = %zu, m = %zu\n", d, m);

    typedef typename ec_type<B>::ECp ECp;
    typedef typename ec_type<B>::ECpe ECpe;

    typedef typename B::G1 G1;
    typedef typename B::G2 G2;

    static constexpr int R = 32;
    static constexpr int C = 5;
    
    auto params = B::read_params(params_file, d, m);
    fclose(params_file);
    print_time(t, "load params");

    auto t_main = t;

    FILE *inputs_file = fopen(input_path, "r");
    auto w_ = load_scalars(m + 1, inputs_file);

    auto ca = load_scalars(d+1, inputs_file);
    auto cb = load_scalars(d+1, inputs_file);
    auto cc = load_scalars(d+1, inputs_file);

    auto r = load_scalars(1, inputs_file);
    
    auto multiplicative_generator = load_scalars(1, inputs_file);
    auto gen_inverse = load_scalars(1, inputs_file);
    auto omega = load_scalars(1, inputs_file);
    auto omega_inverse = load_scalars(1, inputs_file);
    auto sconst = load_scalars(1, inputs_file);
    auto z_inverse = load_scalars(1, inputs_file);
    // var *z_inverse = (var *) malloc (96);
    rewind(inputs_file);
    auto inputs = B::read_input(inputs_file, d, m);
    fclose(inputs_file);
    print_time(t, "load inputs");
    //printf("r: %zu\n", inputs->r);
    const var *w = w_.get();
    printf("w: %zu\n", w);

    FILE *preprocessed_file = fopen(preprocessed_path, "r");

    size_t space = ((m + 1) + R - 1) / R;

    // Previous location for where memory was declared
    //auto A_mults = load_points_affine<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    //auto out_A = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    print_time(t, "load preprocessing");

    auto t_gpu = t;

    cudaStream_t sA, sB1, sB2, sL, sH;

    size_t out_size = space * ECpe::NELTS * ELT_BYTES;

    printf("about to allocate B1\n");

    //auto B1_mults = load_points_affine<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    auto B1_mults = load_points_affine_async<ECp>(sB1, ((1U << C) - 1)*(m + 1), preprocessed_file);
    auto out_B1 = allocate_memory(out_size);

    //printf("allocate B1 affine points and out_B1, out_size: %zu\n", out_size);

    ec_reduce_straus<ECp, C, R>(sB1, out_B1.get(), B1_mults.get(), w, m + 1);
    print_time(t, "gpu launch");
    //printf("ran B1 ec_reduce_straus: B1_mults: %zu\n", B1_mults.get());
    //cudaDeviceSynchronize();

    //cudaStreamSynchronize(sB1);
    //var *host_B1;
    var *host_B1 = (var *) malloc (out_size);
    //printf("address host_B1: %d, out_B1: %zu\n", &host_B1, out_B1.get());
    cudaMemcpyAsync((void **)&host_B1[0], out_B1.get(), out_size, cudaMemcpyDeviceToHost, sB1);
    //cudaDeviceSynchronize();

    printf("about to allocate B2\n");

    //auto B2_mults = load_points_affine<ECpe>(((1U << C) - 1)*(m + 1), preprocessed_file);
    auto B2_mults = load_points_affine_async<ECpe>(sB2, ((1U << C) - 1)*(m + 1), preprocessed_file);
    auto out_B2 = allocate_memory(out_size);

    ec_reduce_straus<ECpe, C, 2*R>(sB2, out_B2.get(), B2_mults.get(), w, m + 1);
    //cudaDeviceSynchronize();

    //cudaStreamSynchronize(sB2);
    var *host_B2 = (var *) malloc (out_size);;
    cudaMemcpyAsync((void **)&host_B2[0], out_B2.get(), out_size, cudaMemcpyDeviceToHost, sB2);
    //cudaDeviceSynchronize();

    printf("about to allocate L\n");

    // auto L_mults = load_points_affine<ECp>(((1U << C) - 1)*(m - 1), preprocessed_file);
    auto L_mults = load_points_affine_async<ECp>(sL, ((1U << C) - 1)*(m - 1), preprocessed_file);
    auto out_L = allocate_memory(out_size);

    ec_reduce_straus<ECp, C, R>(sL, out_L.get(), L_mults.get(), w + (primary_input_size + 1) * ELT_LIMBS, m - 1);
    //cudaDeviceSynchronize();

    //cudaStreamSynchronize(sL);
    var *host_L = (var *) malloc (out_size);;
    cudaMemcpyAsync((void **)&host_L[0], out_L.get(), out_size, cudaMemcpyDeviceToHost, sL);
    //cudaDeviceSynchronize();


    //ec_reduce_straus<ECp, C, R>(sA, out_A.get(), A_mults.get(), w, m + 1);
    //ec_reduce<ECp>(sB1, B1_mults.get(), w, m + 1);
    //ec_reduce_straus<ECp, C, R>(sB1, out_B1.get(), B1_mults.get(), w, m + 1);
    //ec_reduce_straus<ECpe, C, 2*R>(sB2, out_B2.get(), B2_mults.get(), w, m + 1);
    //ec_reduce_straus<ECp, C, R>(sL, out_L.get(), L_mults.get(), w + (primary_input_size + 1) * ELT_LIMBS, m - 1);
    // print_time(t, "gpu launch");
    print_time(t, "gpu 1");
    G1 *evaluation_At = B::multiexp_G1(B::input_w(inputs), B::params_A(params), m + 1);
    //G1 *evaluation_Bt1 = B::multiexp_G1(B::input_w(inputs), B::params_B1(params), m + 1);
    //G2 *evaluation_Bt2 = B::multiexp_G2(B::input_w(inputs), B::params_B2(params), m + 1);
    print_time(t, "cpu msm for A");

    var *a = ca.get();
    var *b = cb.get();
    var *c = cc.get();
    const var *g = multiplicative_generator.get();
    const var *g_inv = gen_inverse.get();
    const var *o = omega.get();
    const var *o_inv = omega_inverse.get();
    const var *sc = sconst.get();
    const var *z_inv = z_inverse.get();
    // Do calculations relating to H on CPU after having set the GPU in
    // motion
    const var *coefficients_for_H =
        compute_H_gpu<B, ECp>(d,
        a, 
        b,
        c,
        g,
        g_inv, 
        o,
        o_inv,
        sc,
        z_inv);

    // var *host_coeff_for_H = (var *) malloc (m * ELT_BYTES);
    // cudaMemcpy((void **)&host_coeff_for_H[0], coefficients_for_H, m * ELT_BYTES, cudaMemcpyDeviceToHost);
    // printf("Address of coeff_for_H is %p\n", (void *)host_coeff_for_H);  
    // printf("Value:  %d\n", *host_coeff_for_H);
    auto H = B::params_H(params);
    // auto coefficients_for_H =
    //     compute_H<B>(d, B::input_ca(inputs), B::input_cb(inputs), B::input_cc(inputs));
    // G1 *evaluation_Ht = B::multiexp_G1(coefficients_for_H, H, d);

    auto H_mults = load_points_affine_async<ECp>(sH, ((1U << C) - 1)*(m - 1), preprocessed_file);
    auto out_H = allocate_memory(out_size);

    ec_reduce_straus<ECp, C, R>(
        sH, 
        out_H.get(), 
        H_mults.get(), 
        coefficients_for_H,  
        d);
    
    var *host_H = (var *) malloc (out_size);;
    cudaMemcpyAsync((void **)&host_H[0], out_H.get(), out_size, cudaMemcpyDeviceToHost, sH);

    fclose(preprocessed_file);

    //cudaDeviceSynchronize();
    //cudaStreamSynchronize(sA);
    //G1 *evaluation_At = B::read_pt_ECp(out_A.get());

    //cudaStreamSynchronize(sB1);
    //G1 *evaluation_Bt1 = B::read_pt_ECp(out_B1.get());

    //cudaStreamSynchronize(sB2);
    //G2 *evaluation_Bt2 = B::read_pt_ECpe(out_B2.get());

    // cudaStreamSynchronize(sL);
    // G1 *evaluation_Lt = B::read_pt_ECp(out_L.get());

    printf("host_B1: %zu\n", host_B1);
    cudaStreamSynchronize(sB1);
    G1 *evaluation_Bt1 = B::read_pt_ECp(host_B1);
    cudaStreamSynchronize(sB2);
    G2 *evaluation_Bt2 = B::read_pt_ECpe(host_B2);
    cudaStreamSynchronize(sL);
    G1 *evaluation_Lt = B::read_pt_ECp(host_L);
    cudaStreamSynchronize(sH);
    G1 *evaluation_Ht = B::read_pt_ECp(host_H);

    print_time(t_gpu, "gpu e2e");

    auto scaled_Bt1 = B::G1_scale(B::input_r(inputs), evaluation_Bt1);
    auto Lt1_plus_scaled_Bt1 = B::G1_add(evaluation_Lt, scaled_Bt1);
    auto final_C = B::G1_add(evaluation_Ht, Lt1_plus_scaled_Bt1);

    print_time(t, "cpu 2");
    B::print_G1(evaluation_Bt1);
    B::print_G2(evaluation_Bt2);
    B::print_G1(evaluation_Lt);
    B::print_G1(evaluation_Ht);

    B::groth16_output_write(evaluation_At, evaluation_Bt2, final_C, output_path);

    print_time(t, "store");

    print_time(t_main, "Total time from input to output: ");

    //cudaStreamDestroy(sA);
    cudaStreamDestroy(sB1);
    cudaStreamDestroy(sB2);
    cudaStreamDestroy(sL);

    free(host_B1);
    free(host_B2);
    free(host_L);

    B::delete_vector_G1(H);

    B::delete_G1(evaluation_At);
    B::delete_G1(evaluation_Bt1);
    B::delete_G2(evaluation_Bt2);
    B::delete_G1(evaluation_Ht);
    B::delete_G1(evaluation_Lt);
    B::delete_G1(scaled_Bt1);
    B::delete_G1(Lt1_plus_scaled_Bt1);
    // B::delete_vector_Fr(coefficients_for_H);
    B::delete_groth16_input(inputs);
    B::delete_groth16_params(params);

    print_time(t, "cleanup");
    print_time(beginning, "Total runtime (incl. file reads)");
}

template <typename B>
void run_prover(
        const char *params_path,
        const char *input_path,
        const char *output_path,
        const char *preprocessed_path)
{
    B::init_public_params();

    size_t primary_input_size = 1;

    auto beginning = now();
    auto t = beginning;

    FILE *params_file = fopen(params_path, "r");
    size_t d = read_size_t(params_file);
    size_t m = read_size_t(params_file);
    rewind(params_file);

    printf("d = %zu, m = %zu\n", d, m);

    typedef typename ec_type<B>::ECp ECp;
    typedef typename ec_type<B>::ECpe ECpe;

    typedef typename B::G1 G1;
    typedef typename B::G2 G2;

    static constexpr int R = 32;
    static constexpr int C = 5;
    FILE *preprocessed_file = fopen(preprocessed_path, "r");

    size_t space = ((m + 1) + R - 1) / R;

    //auto A_mults = load_points_affine<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    //auto out_A = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    auto B1_mults = load_points_affine<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    auto out_B1 = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    auto B2_mults = load_points_affine<ECpe>(((1U << C) - 1)*(m + 1), preprocessed_file);
    auto out_B2 = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    auto L_mults = load_points_affine<ECp>(((1U << C) - 1)*(m - 1), preprocessed_file);
    auto out_L = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    fclose(preprocessed_file);

    print_time(t, "load preprocessing");

    auto params = B::read_params(params_file, d, m);
    fclose(params_file);
    print_time(t, "load params");

    auto t_main = t;

    FILE *inputs_file = fopen(input_path, "r");
    auto w_ = load_scalars(m + 1, inputs_file);
    rewind(inputs_file);
    auto inputs = B::read_input(inputs_file, d, m);
    fclose(inputs_file);
    print_time(t, "load inputs");

    const var *w = w_.get();

    auto t_gpu = t;

    cudaStream_t sA, sB1, sB2, sL;

    //ec_reduce_straus<ECp, C, R>(sA, out_A.get(), A_mults.get(), w, m + 1);
    ec_reduce_straus<ECp, C, R>(sB1, out_B1.get(), B1_mults.get(), w, m + 1);
    ec_reduce_straus<ECpe, C, 2*R>(sB2, out_B2.get(), B2_mults.get(), w, m + 1);
    ec_reduce_straus<ECp, C, R>(sL, out_L.get(), L_mults.get(), w + (primary_input_size + 1) * ELT_LIMBS, m - 1);
    print_time(t, "gpu launch");

    G1 *evaluation_At = B::multiexp_G1(B::input_w(inputs), B::params_A(params), m + 1);
    //G1 *evaluation_Bt1 = B::multiexp_G1(B::input_w(inputs), B::params_B1(params), m + 1);
    //G2 *evaluation_Bt2 = B::multiexp_G2(B::input_w(inputs), B::params_B2(params), m + 1);

    // Do calculations relating to H on CPU after having set the GPU in
    // motion
    auto H = B::params_H(params);
    auto coefficients_for_H =
        compute_H<B>(d, B::input_ca(inputs), B::input_cb(inputs), B::input_cc(inputs));
    G1 *evaluation_Ht = B::multiexp_G1(coefficients_for_H, H, d);

    print_time(t, "cpu 1");

    cudaDeviceSynchronize();
    //cudaStreamSynchronize(sA);
    //G1 *evaluation_At = B::read_pt_ECp(out_A.get());

    cudaStreamSynchronize(sB1);
    G1 *evaluation_Bt1 = B::read_pt_ECp(out_B1.get());

    cudaStreamSynchronize(sB2);
    G2 *evaluation_Bt2 = B::read_pt_ECpe(out_B2.get());

    cudaStreamSynchronize(sL);
    G1 *evaluation_Lt = B::read_pt_ECp(out_L.get());

    print_time(t_gpu, "gpu e2e");

    auto scaled_Bt1 = B::G1_scale(B::input_r(inputs), evaluation_Bt1);
    auto Lt1_plus_scaled_Bt1 = B::G1_add(evaluation_Lt, scaled_Bt1);
    auto final_C = B::G1_add(evaluation_Ht, Lt1_plus_scaled_Bt1);

    print_time(t, "cpu 2");
    B::print_G1(evaluation_Bt1);
    B::print_G2(evaluation_Bt2);
    B::print_G1(evaluation_Lt);
    
    B::groth16_output_write(evaluation_At, evaluation_Bt2, final_C, output_path);

    print_time(t, "store");

    print_time(t_main, "Total time from input to output: ");

    //cudaStreamDestroy(sA);
    cudaStreamDestroy(sB1);
    cudaStreamDestroy(sB2);
    cudaStreamDestroy(sL);

    B::delete_vector_G1(H);

    B::delete_G1(evaluation_At);
    B::delete_G1(evaluation_Bt1);
    B::delete_G2(evaluation_Bt2);
    B::delete_G1(evaluation_Ht);
    B::delete_G1(evaluation_Lt);
    B::delete_G1(scaled_Bt1);
    B::delete_G1(Lt1_plus_scaled_Bt1);
    B::delete_vector_Fr(coefficients_for_H);
    B::delete_groth16_input(inputs);
    B::delete_groth16_params(params);

    print_time(t, "cleanup");
    print_time(beginning, "Total runtime (incl. file reads)");
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  std::string curve(argv[1]);
  std::string mode(argv[2]);

  const char *params_path = argv[3];

  if (mode == "compute") {
      const char *input_path = argv[4];
      const char *output_path = argv[5];
      std::string gpu(argv[6]);
      if (gpu == "gpu-fft") {
        if (curve == "MNT4753") {
            run_prover_gpu<mnt4753_libsnark>(params_path, input_path, output_path, "MNT4753_preprocessed");
        } else if (curve == "MNT6753") {
            // Temporary for testing
            run_prover_gpu<mnt6753_libsnark>(params_path, input_path, output_path, "MNT6753_preprocessed");
        }
      } else if (gpu == "gpu-orig") {
        if (curve == "MNT4753") {
          run_prover<mnt4753_libsnark>(params_path, input_path, output_path, "MNT4753_preprocessed");
        } else if (curve == "MNT6753") {
          // Temporary for testing
          run_prover<mnt6753_libsnark>(params_path, input_path, output_path, "MNT6753_preprocessed");
        }
      }

  } else if (mode == "preprocess") {
#if 0
      if (curve == "MNT4753") {
          run_preprocess<mnt4753_libsnark>(params_path);
      } else if (curve == "MNT6753") {
          run_preprocess<mnt4753_libsnark>(params_path);
      }
#endif
  }

  return 0;
}
