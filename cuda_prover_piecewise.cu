#include <string>
#include <chrono>
#include <memory>
#include <inttypes.h>
#include <numeric>
#include <vector>

#define NDEBUG 1

#include <prover_reference_functions.hpp>
#include "multiexp/reduce.cu"
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
// #include <libff/algebra/curves/mnt753/mnt4753/mnt4753_init.hpp>
// #include <libff/algebra/curves/mnt753/mnt6753/mnt6753_init.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>
#include <libfqfft/tools/exceptions.hpp>
#include <libfqfft/evaluation_domain/evaluation_domain.hpp>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
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

void *
load_scalars_async_host(size_t n, FILE *inputs)
{
    static constexpr size_t scalar_bytes = ELT_BYTES;
    size_t total_bytes = n * scalar_bytes;
    printf("total scalar bytes host alloc: %zu\n", total_bytes);

    // void *scalars_buffer = (void *) malloc (total_bytes);
    void *scalars_buffer;
    gpuErrchk( cudaMallocHost(&scalars_buffer, total_bytes) );
    if (fread(scalars_buffer, total_bytes, 1, inputs) < 1) {
        fprintf(stderr, "Failed to read scalars\n");
        abort();
    }

    return scalars_buffer;
}

template< typename EC >
void *
load_points_affine_host(size_t n, FILE *inputs)
{
    typedef typename EC::field_type FF;

    static constexpr size_t coord_bytes = FF::DEGREE * ELT_BYTES;
    static constexpr size_t aff_pt_bytes = 2 * coord_bytes;

    size_t total_aff_bytes = n * aff_pt_bytes;
    printf("total affine bytes: %zu\n", total_aff_bytes);
    // auto mem = allocate_memory(total_aff_bytes, 1);

    // void *aff_bytes_buffer = (void *) malloc (total_aff_bytes);
    void *aff_bytes_buffer;
    gpuErrchk( cudaMallocHost((void **)&aff_bytes_buffer, total_aff_bytes) );
    if (fread(aff_bytes_buffer, total_aff_bytes, 1, inputs) < 1) {
        fprintf(stderr, "Failed to read all curve poinst\n");
        abort();
    }
    // printf("aff_bytes_buffer: %d\n", (int *)aff_bytes_buffer + (total_aff_bytes - 96));

    return aff_bytes_buffer;
}

template<typename EC>
size_t
get_aff_total_bytes(size_t n) 
{
    typedef typename EC::field_type FF;

    static constexpr size_t coord_bytes = FF::DEGREE * ELT_BYTES;
    static constexpr size_t aff_pt_bytes = 2 * coord_bytes;
    size_t total_aff_bytes = n * aff_pt_bytes;
    // printf("total affine bytes for %ld points: %zu\n", n, total_aff_bytes);
    return total_aff_bytes;
}

template <typename B>
void run_prover(
        const char *params_path,
        const char *input_path,
        const char *output_path,
        const char *preprocessed_path)
{
    B::init_public_params();

    cudaFree(0);
    size_t primary_input_size = 1;

    const size_t CHUNKS = 4;

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
    void *w_host = load_scalars_async_host(m + 1, inputs_file);
    // auto w_ = load_scalars_async(m + 1, inputs_file);
    rewind(inputs_file);
    void *w_host2 = load_scalars_async_host(m + 1, inputs_file);
    rewind(inputs_file);
    void *w_host3 = load_scalars_async_host(m + 1, inputs_file);
    rewind(inputs_file);
    auto inputs = B::read_input(inputs_file, d, m);
    fclose(inputs_file);
    print_time(t, "load inputs");

    // Used before switching to async memcpy-ing and no unified memory
    // const var *w = w_.get();
    // printf("w: %zu\n", w);

    FILE *preprocessed_file = fopen(preprocessed_path, "r");

    size_t space = ((m + 1) + R - 1) / R;

    print_time(t, "load preprocessing");

    auto t_gpu = t;

    cudaStream_t sA, sB1, sB2, sL;

    // cudaStream_t sA[CHUNKS], sB1[CHUNKS], sB2[CHUNKS], sL[CHUNKS];

    size_t out_size = space * ECpe::NELTS * ELT_BYTES;
    size_t w_size = (m+1)*ELT_BYTES;
    printf("w_size: %ld\n", w_size);
    printf("out_size: %ld\n", out_size);
    size_t B1_mults_size = get_aff_total_bytes<ECp>(((1U << C) - 1)*(m + 1));
    size_t B2_mults_size = get_aff_total_bytes<ECpe>(((1U << C) - 1)*(m + 1));
    size_t L_mults_size = get_aff_total_bytes<ECp>(((1U << C) - 1)*(m - 1));
    printf("B1_mults_size: %ld\n", B1_mults_size);
    printf("B2_mults_size: %ld\n", B2_mults_size);
    printf("L_mults_size: %ld\n", L_mults_size);

    size_t m_chunked = m / CHUNKS;
    printf("m / CHUNKS: %ld\n", m_chunked);
    size_t B_m_chunked = m_chunked;
    size_t L_m_chunked = m_chunked;

    int B_m_chunks[CHUNKS];
    int L_m_chunks[CHUNKS];
    printf("about to allocate out ptrs\n");

    // Previous location for where memory was declared
    // auto A_mults = load_points_affine_async<ECp>(sA, ((1U << C) - 1)*(m + 1), preprocessed_file);
    // auto out_A = allocate_memory(out_size);


    void *B1_mults_host_chunked[CHUNKS];
    void *B2_mults_host_chunked[CHUNKS];
    void *L_mults_host_chunked[CHUNKS];

    // Each of the CHUNK'd arrays is an output of a multiexponentation 
    var_ptr out_B1[CHUNKS];
    var_ptr out_B2[CHUNKS];
    var_ptr out_L[CHUNKS];

    var *host_B1[CHUNKS];
    var *host_B2[CHUNKS];
    var *host_L[CHUNKS];

    printf("about to allocate B1\n");

    // void *B1_mults_host = load_points_affine_host<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    
    for (size_t chunk = 0; chunk < CHUNKS; chunk++) {
        if (chunk == CHUNKS - 1) {
            B_m_chunks[chunk] = m_chunked + 1;
            L_m_chunks[chunk] = m_chunked - 1;
        } else {
            B_m_chunks[chunk] = m_chunked;
            L_m_chunks[chunk] = m_chunked;
        }
        size_t chunk_offset = get_aff_total_bytes<ECp>(((1U << C) - 1)*B_m_chunks[0]*chunk);
        size_t chunk_size = get_aff_total_bytes<ECp>(((1U << C) - 1)*B_m_chunks[chunk]);
        void *source = load_points_affine_host<ECp>(((1U << C) - 1)*B_m_chunks[chunk], preprocessed_file);
        // B1_mults_host_chunked[chunk] = source;
        gpuErrchk( cudaMallocHost((void **)&B1_mults_host_chunked[chunk], chunk_size) );
        std::memcpy(B1_mults_host_chunked[chunk], source, chunk_size);
        printf("chunk, B1_mults_host_chunked[%ld]: %p\n", chunk, B1_mults_host_chunked[chunk]);
        printf("chunk_offset: %ld, chunk_size: %p\n", chunk_offset, chunk_size);

        out_B1[chunk] = allocate_memory(out_size, 1);
        printf("out_B1[%d]: %p\n", chunk, out_B1[chunk].get());

        out_B2[chunk] = allocate_memory(out_size, 1);
        out_L[chunk] = allocate_memory(out_size, 1);

        cudaMallocHost((void **)&host_B1[chunk], out_size);
        // printf("host_B1: %p\n", host_B1[i]);

        cudaMallocHost((void **)&host_B2[chunk], out_size);
        // printf("host_B2: %p\n", host_B2[i]);

        cudaMallocHost((void **)&host_L[chunk], out_size);
    }
    // printf("B1_mults_host: %p\n", B1_mults_host);

    printf("about to allocate B2\n");
    void *B2_mults_host = load_points_affine_host<ECpe>(((1U << C) - 1)*(m + 1), preprocessed_file);

    printf("about to allocate L\n");
    void *L_mults_host = load_points_affine_host<ECp>(((1U << C) - 1)*(m - 1), preprocessed_file);

    fclose(preprocessed_file);

    // printf("about to allocate A\n");
    // ec_reduce_straus<ECp, C, R>(sA, out_A.get(), A_mults.get(), w, m + 1);
    // var *host_A = (var *) malloc (out_size);
    // cudaMemcpyAsync((void **)&host_A[0], out_A.get(), out_size, cudaMemcpyDeviceToHost, sA);
    
    cudaStreamCreateWithFlags(&sB1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&sB2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&sL, cudaStreamNonBlocking);

    // TODO: consider utilizing shared memory for the witness
    // auto w1 = allocate_memory(w_size, 1);
    // auto w2 = allocate_memory(w_size, 1);
    // auto w3 = allocate_memory(w_size, 1);

    size_t w_size_chunked = w_size / CHUNKS;
    printf("w_size / CHUNKS: %ld\n", w_size_chunked);
    size_t B1_mults_size_chunked = B1_mults_size / CHUNKS;
    printf("B1_mults_size / CHUNKS: %ld\n", B1_mults_size_chunked);
    size_t B2_mults_size_chunked = B2_mults_size / CHUNKS;
    printf("B2_mults_size / CHUNKS: %ld\n", B2_mults_size_chunked);
    size_t L_mults_size_chunked = L_mults_size / CHUNKS;
    printf("L_mults_size / CHUNKS: %ld\n", L_mults_size_chunked);

    // TODO: do the same thing with the offsets as the chunks amounts so that we don't have to encapsulate all the CUDA calls in big IF statements
    // Doing this will clean up a lot of the repeated statements seen below
    // size_t w_offset_B_m = 
    printf("ELT_BYTES: %ld\n", ELT_BYTES);

    for (size_t i = 0; i < CHUNKS; i++) {
        if (i == CHUNKS - 1) {
            B_m_chunked = m_chunked + 1;
            // printf("(m + 1) / CHUNKS: %ld\n", B_m_chunked);
            L_m_chunked = m_chunked - 1;
            // printf("(m - 1) / CHUNKS: %ld\n", L_m_chunked);
        }

        auto w1 = allocate_memory(B_m_chunked * ELT_BYTES, 1);
        auto w2 = allocate_memory(B_m_chunked * ELT_BYTES, 1);
        auto w3 = allocate_memory(L_m_chunked * ELT_BYTES, 1);

        printf("w1 and w2 size: %ld\n", B_m_chunked * ELT_BYTES);
        printf("w3 size: %ld\n", L_m_chunked * ELT_BYTES);

        auto B1_mults = allocate_memory(get_aff_total_bytes<ECp>(((1U << C) - 1)*B_m_chunked), 1);
        auto B2_mults = allocate_memory(get_aff_total_bytes<ECpe>(((1U << C) - 1)*B_m_chunked), 1);
        auto L_mults = allocate_memory(get_aff_total_bytes<ECp>(((1U << C) - 1)*L_m_chunked), 1);

        // Need to do this because we are working with var * types that index 8 bytes at a time
        // size_t out_size_scaled = (i * out_size) / 8;

        // TODO: check how we chunk through the multiples and witness, possibly have values overlapping when memcpy'ing causing errors

        printf("i * (B_m_chunked) * ELT_BYTES: %p\n", (i * (B_m_chunked) * ELT_BYTES));
        printf("w_host + i * (B_m_chunked) * ELT_BYTES: %p\n", w_host + (i * (B_m_chunked) * ELT_BYTES));
        printf("B_m_chunked * ELT_BYTES: %ld\n", B_m_chunked * ELT_BYTES);
        // cudaDeviceSynchronize();

        if (i == CHUNKS - 1) {
            printf("get_aff_total_bytes<ECp>(((1U << C) - 1)*B_m_chunks[i]): %ld\n", get_aff_total_bytes<ECp>(((1U << C) - 1)*B_m_chunks[i]) );
            // gpuErrchk( 
            //     cudaMemcpyAsync(B1_mults.get(), 
            //     B1_mults_host + get_aff_total_bytes<ECp>(((1U << C) - 1)* i * (B_m_chunked - 1)), 
            //     get_aff_total_bytes<ECp>(((1U << C) - 1)*B_m_chunked), 
            //     cudaMemcpyHostToDevice, 
            //     sB1) );
            printf("B1_mults_host_chunked[%ld]: %p\n", i, B1_mults_host_chunked[i]);
           
            gpuErrchk( 
                cudaMemcpyAsync(B1_mults.get(), 
                B1_mults_host_chunked[i], 
                get_aff_total_bytes<ECp>(((1U << C) - 1)*B_m_chunks[i]), 
                cudaMemcpyHostToDevice, 
                sB1) );
            printf("B1_mults.get(): %p\n", B1_mults.get());

            gpuErrchk( 
                cudaMemcpyAsync(B2_mults.get(), 
                B2_mults_host + get_aff_total_bytes<ECpe>(((1U << C) - 1)* i * (B_m_chunked - 1)), 
                get_aff_total_bytes<ECpe>(((1U << C) - 1)*B_m_chunked), 
                cudaMemcpyHostToDevice, 
                sB2) );

            gpuErrchk( 
                cudaMemcpyAsync(L_mults.get(), 
                L_mults_host + get_aff_total_bytes<ECp>(((1U << C) - 1)* i * (L_m_chunked + 1)), 
                get_aff_total_bytes<ECp>(((1U << C) - 1)*L_m_chunked), 
                cudaMemcpyHostToDevice, 
                sL) );
            
            printf("i * (B_m_chunked - 1) * ELT_BYTES: %p\n", (i * (B_m_chunked - 1) * ELT_BYTES));
            printf("w_host + (i * (B_m_chunked - 1) * ELT_BYTES): %p\n", w_host + (i * (B_m_chunked - 1) * ELT_BYTES));
            gpuErrchk( cudaMemcpyAsync(w1.get(), ((char *)w_host) + (i * (B_m_chunked - 1) * ELT_BYTES), B_m_chunked * ELT_BYTES, cudaMemcpyHostToDevice, sB1) ); 
            gpuErrchk( cudaMemcpyAsync(w2.get(), ((char *)w_host2)  + (i * (B_m_chunked - 1) * ELT_BYTES), B_m_chunked * ELT_BYTES, cudaMemcpyHostToDevice, sB2) ); 

            gpuErrchk( cudaMemcpyAsync(w3.get(), ((char *)w_host3) + ((2+(i * (L_m_chunked + 1))) * ELT_BYTES), L_m_chunked * ELT_BYTES, cudaMemcpyHostToDevice, sL) ); 

        } else {
            // gpuErrchk( 
            //     cudaMemcpyAsync(B1_mults.get(), 
            //     B1_mults_host + get_aff_total_bytes<ECp>(((1U << C) - 1)* i * B_m_chunked), 
            //     get_aff_total_bytes<ECp>(((1U << C) - 1)*B_m_chunked), 
            //     cudaMemcpyHostToDevice, 
            //     sB1) );
            printf("B1_mults_host_chunked[%ld]: %p\n", i, B1_mults_host_chunked[i]);
            printf("get_aff_total_bytes<ECp>(((1U << C) - 1)* i * B_m_chunks[i]): %ld\n", get_aff_total_bytes<ECp>(((1U << C) - 1)* i * B_m_chunks[i]) );
            // printf("B1_mults_host_chunked[%ld]: %p\n", i, B1_mults_host_chunked[get_aff_total_bytes<ECp>(((1U << C) - 1)* i * B_m_chunks[i])]);

            gpuErrchk( 
                cudaMemcpyAsync(B1_mults.get(), 
                B1_mults_host_chunked[i], 
                get_aff_total_bytes<ECp>(((1U << C) - 1)*B_m_chunks[i]), 
                cudaMemcpyHostToDevice, 
                sB1) );
            printf("B1_mults.get(): %p\n", B1_mults.get());

            gpuErrchk( 
                cudaMemcpyAsync(B2_mults.get(), 
                B2_mults_host + get_aff_total_bytes<ECpe>(((1U << C) - 1)* i * B_m_chunked), 
                get_aff_total_bytes<ECpe>(((1U << C) - 1)*B_m_chunked), 
                cudaMemcpyHostToDevice, 
                sB2) );

            gpuErrchk( 
                cudaMemcpyAsync(L_mults.get(), 
                L_mults_host + get_aff_total_bytes<ECp>(((1U << C) - 1)* i * L_m_chunked), 
                get_aff_total_bytes<ECp>(((1U << C) - 1)*L_m_chunked), 
                cudaMemcpyHostToDevice, 
                sL) );

            gpuErrchk( cudaMemcpyAsync(w1.get(), ((char *)w_host) + (i * (B_m_chunked) * ELT_BYTES), B_m_chunked * ELT_BYTES, cudaMemcpyHostToDevice, sB1) ); 
            gpuErrchk( cudaMemcpyAsync(w2.get(), ((char *)w_host2) + (i * (B_m_chunked) * ELT_BYTES), B_m_chunked * ELT_BYTES, cudaMemcpyHostToDevice, sB2) );

            gpuErrchk( cudaMemcpyAsync(w3.get(), ((char *)w_host3) + ((2+(i * L_m_chunked)) * ELT_BYTES), L_m_chunked * ELT_BYTES, cudaMemcpyHostToDevice, sL) ); 

        }

        cudaDeviceSynchronize();

        ec_reduce_straus<ECp, C, R>(sB1, out_B1[i].get(), B1_mults.get(), w1.get(), B_m_chunked);
        // ec_reduce<ECp>(sB1, )
        printf("out of ec reduce B1, on host\n");
        printf("i * B1_mults_size_chunked: %ld\n", i * B1_mults_size_chunked);
        // printf("out_size_scaled: %ld\n", out_size_scaled);

        // gpuErrchk( cudaMemcpyAsync(B2_mults.get(), B2_mults_host + get_aff_total_bytes<ECpe>(i * B_m_chunked), get_aff_total_bytes<ECpe>(B_m_chunked), cudaMemcpyHostToDevice, sB2) );
        ec_reduce_straus<ECpe, C, 2*R>(sB2, out_B2[i].get(), B2_mults.get(), w2.get(), B_m_chunked);
        printf("out of ec reduce B2, on host\n");

        // gpuErrchk( cudaMemcpyAsync(L_mults.get(), L_mults_host + (i * L_m_chunked * 2) * ELT_BYTES, 2 * L_m_chunked * ELT_BYTES, cudaMemcpyHostToDevice, sL) );
        printf("w_host3: %p\n", w_host3);
        printf("w_host3 + (2+(i * L_m_chunked)) * ELT_BYTES: %p\n", w_host3 + (2+(i * L_m_chunked)) * ELT_BYTES);
        // NOTE: it is only + (2 * ELT_LIMBS) as w3 is a var * that jumps by 64 bits. 12 * 64 = 768 bit element
        printf("w3.get(): %p\n", w3.get());
        printf("w3.get() + (primary_input_size + 1) * ELT_LIMBS: %p\n", w3.get() + (primary_input_size + 1) * ELT_LIMBS);
        ec_reduce_straus<ECp, C, R>(sL, out_L[i].get(), L_mults.get(), w3.get(), L_m_chunked);
        printf("out of ec reduce L, on host\n");

        cudaDeviceSynchronize();

        printf("out_B1[%d].get(): %p\n", i, out_B1[i].get()); 
        gpuErrchk( cudaMemcpyAsync(host_B1[i], out_B1[i].get(), out_size, cudaMemcpyDeviceToHost, sB1) );
        printf("initiated B1 copy to host\n");

        printf("out_B2[%d].get(): %p\n", i, out_B2[i].get()); 
        gpuErrchk( cudaMemcpyAsync(host_B2[i], out_B2[i].get(), out_size, cudaMemcpyDeviceToHost, sB2) );
        printf("initiated B2 copy to host\n");

        gpuErrchk( cudaMemcpyAsync(host_L[i], out_L[i].get(), out_size, cudaMemcpyDeviceToHost, sL) );
        printf("initiated L copy to host\n");
    }

    print_time(t, "gpu launch");

    G1 *evaluation_At = B::multiexp_G1(B::input_w(inputs), B::params_A(params), m + 1);

    // Do calculations relating to H on CPU after having set the GPU in
    // motion
    auto H = B::params_H(params);
    auto coefficients_for_H =
        compute_H<B>(d, B::input_ca(inputs), B::input_cb(inputs), B::input_cc(inputs));

    G1 *evaluation_Ht = B::multiexp_G1(coefficients_for_H, H, d);

    print_time(t, "cpu 1");

    // cudaDeviceSynchronize();
    //cudaStreamSynchronize(sA);
    //G1 *evaluation_At = B::read_pt_ECp(out_A.get());

    cudaStreamSynchronize(sB1);
    printf("synchronized sB1\n");
    printf("host_B1: %" PRIu64 "\n", *(host_B1[0]));
    // G1 *evaluation_Bt1 = B::read_pt_ECp(host_B1);
    // G1 *B1_evaluations[CHUNKS];
    std::vector<G1*> B1_evaluations(CHUNKS);

    for (size_t i = 0; i < CHUNKS; i++) {
        B1_evaluations[i] = B::read_pt_ECp(host_B1[i]);
        B::print_G1(B1_evaluations[i]);
    }

    auto binary_op = [](G1 *p1, G1 *p2){return B::G1_add(p1, p2); };
    G1 *final_B1_host = std::accumulate(B1_evaluations.begin() + 1, B1_evaluations.end(), B1_evaluations[0], binary_op);
    printf("final_B1_host:\n");
    B::print_G1(final_B1_host);

    cudaStreamSynchronize(sB2);
    printf("synchronized sB2\n");
    printf("host_B2: %" PRIu64 "\n", *(host_B2[0]));
    // G2 *evaluation_Bt2 = B::read_pt_ECpe(host_B2);
    G2 *B2_evaluations[CHUNKS];
    for (size_t i = 0; i < CHUNKS; i++) {
        B2_evaluations[i] = B::read_pt_ECpe(host_B2[i]);
    }

    cudaStreamSynchronize(sL);
    printf("synchronized sL\n");
    printf("host_L: %" PRIu64 "\n", *(host_L[0]));
    // G1 *evaluation_Lt = B::read_pt_ECp(host_L);
    G1 *L_evaluations[CHUNKS];
    for (size_t i = 0; i < CHUNKS; i++) {
        L_evaluations[i] = B::read_pt_ECp(host_L[i]);
    }
    
    G1 *evaluation_Bt1_sum = B1_evaluations[0];
    G2 *evaluation_Bt2 = B2_evaluations[0];
    G1 *evaluation_Lt = L_evaluations[0];
    for (size_t i = 1; i < CHUNKS; i++) {
        printf("before addition:\n");
        B::print_G1(evaluation_Bt1_sum);
        evaluation_Bt1_sum = B::G1_add(evaluation_Bt1_sum, B1_evaluations[i]);
        printf("after addition:\n");
        B::print_G1(evaluation_Bt1_sum);
        evaluation_Bt2 = B::G2_add(evaluation_Bt2, B2_evaluations[i]);
        // B::print_G2(evaluation_Bt2);
        evaluation_Lt = B::G1_add(evaluation_Lt, L_evaluations[i]);
        // B::print_G1(evaluation_Lt);
    }
    G1 *evaluation_Bt1 = evaluation_Bt1_sum;

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

    for (size_t chunk = 0; chunk < CHUNKS; chunk++) {
        cudaFreeHost(B1_mults_host_chunked[chunk]);
        cudaFreeHost(host_B1[chunk]);
        cudaFreeHost(host_B2[chunk]);
        cudaFreeHost(host_L[chunk]);
    }
    // cudaFreeHost(B1_mults_host_chunked);
    cudaFreeHost(B2_mults_host);
    cudaFreeHost(L_mults_host);
    cudaFreeHost(w_host);
    cudaFreeHost(w_host2);
    cudaFreeHost(w_host3);
    // cudaFreeHost(host_B1);
    // cudaFreeHost(host_B2);
    // cudaFreeHost(host_L);

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
