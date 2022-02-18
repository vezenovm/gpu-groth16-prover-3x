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

//#include "fft.cu"
// This is where all the FFTs happen

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
    printf("w: %zu\n", w);

    FILE *preprocessed_file = fopen(preprocessed_path, "r");

    size_t space = ((m + 1) + R - 1) / R;

    // Previous location for where memory was declared
    //auto A_mults = load_points_affine<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    //auto out_A = allocate_memory(space * ECpe::NELTS * ELT_BYTES);


    print_time(t, "load preprocessing");

    auto t_gpu = t;

    cudaStream_t sA, sB1, sB2, sL;

    size_t out_size = space * ECpe::NELTS * ELT_BYTES;

    printf("about to allocate B1\n");

    auto B1_mults = load_points_affine<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    auto out_B1 = allocate_memory(out_size);

    //printf("allocate B1 affine points and out_B1, out_size: %zu\n", out_size);

    ec_reduce_straus<ECp, C, R>(sB1, out_B1.get(), B1_mults.get(), w, m + 1);

    //printf("ran B1 ec_reduce_straus: B1_mults: %zu\n", B1_mults.get());
    //cudaDeviceSynchronize();

    //cudaStreamSynchronize(sB1);
    //var *host_B1;
    var *host_B1 = (var *) malloc (out_size);
    //printf("address host_B1: %d, out_B1: %zu\n", &host_B1, out_B1.get());
    cudaMemcpyAsync((void **)&host_B1[0], out_B1.get(), out_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("about to allocate B2\n");

    auto B2_mults = load_points_affine<ECpe>(((1U << C) - 1)*(m + 1), preprocessed_file);
    auto out_B2 = allocate_memory(out_size);

    ec_reduce_straus<ECpe, C, 2*R>(sB2, out_B2.get(), B2_mults.get(), w, m + 1);
    //cudaDeviceSynchronize();

    //cudaStreamSynchronize(sB2);
    var *host_B2 = (var *) malloc (out_size);;
    cudaMemcpyAsync((void **)&host_B2[0], out_B2.get(), out_size, cudaMemcpyDeviceToHost, sB2);
    cudaDeviceSynchronize();

    printf("about to allocate L\n");

    auto L_mults = load_points_affine<ECp>(((1U << C) - 1)*(m - 1), preprocessed_file);
    auto out_L = allocate_memory(out_size);

    ec_reduce_straus<ECp, C, R>(sL, out_L.get(), L_mults.get(), w + (primary_input_size + 1) * ELT_LIMBS, m - 1);
    //cudaDeviceSynchronize();

    //cudaStreamSynchronize(sL);
    var *host_L = (var *) malloc (out_size);;
    cudaMemcpyAsync((void **)&host_L[0], out_L.get(), out_size, cudaMemcpyDeviceToHost, sL);
    cudaDeviceSynchronize();

    fclose(preprocessed_file);

    //ec_reduce_straus<ECp, C, R>(sA, out_A.get(), A_mults.get(), w, m + 1);
    //ec_reduce<ECp>(sB1, B1_mults.get(), w, m + 1);
    //ec_reduce_straus<ECp, C, R>(sB1, out_B1.get(), B1_mults.get(), w, m + 1);
    //ec_reduce_straus<ECpe, C, 2*R>(sB2, out_B2.get(), B2_mults.get(), w, m + 1);
    //ec_reduce_straus<ECp, C, R>(sL, out_L.get(), L_mults.get(), w + (primary_input_size + 1) * ELT_LIMBS, m - 1);
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

    //cudaDeviceSynchronize();
    //cudaStreamSynchronize(sA);
    //G1 *evaluation_At = B::read_pt_ECp(out_A.get());

    //cudaStreamSynchronize(sB1);
    //G1 *evaluation_Bt1 = B::read_pt_ECp(out_B1.get());

    //cudaStreamSynchronize(sB2);
    //G2 *evaluation_Bt2 = B::read_pt_ECpe(out_B2.get());

    //cudaStreamSynchronize(sL);
    //G1 *evaluation_Lt = B::read_pt_ECp(out_L.get());
    printf("host_B1: %zu\n", host_B1);
    //cudaStreamSynchronize(sB1);
    G1 *evaluation_Bt1 = B::read_pt_ECp(host_B1);
    //cudaStreamSynchronize(sB2);
    G2 *evaluation_Bt2 = B::read_pt_ECpe(host_B2);
    //cudaStreamSynchronize(sL);
    G1 *evaluation_Lt = B::read_pt_ECp(host_L);

    print_time(t_gpu, "gpu e2e");

    auto scaled_Bt1 = B::G1_scale(B::input_r(inputs), evaluation_Bt1);
    auto Lt1_plus_scaled_Bt1 = B::G1_add(evaluation_Lt, scaled_Bt1);
    auto final_C = B::G1_add(evaluation_Ht, Lt1_plus_scaled_Bt1);

    print_time(t, "cpu 2");

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

      if (curve == "MNT4753") {
          run_prover<mnt4753_libsnark>(params_path, input_path, output_path, "MNT4753_preprocessed");
      } else if (curve == "MNT6753") {
          // Temporary for testing
          run_prover<mnt6753_libsnark>(params_path, input_path, output_path, "MNT6753_preprocessed");
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
