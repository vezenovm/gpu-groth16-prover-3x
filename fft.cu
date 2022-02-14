//#include <libfqfft/evaluation_domain/domains/basic_radix2_domain_aux.hpp>
#include <prover_reference_functions.hpp>

// static constexpr size_t threads_per_block = 256;
// static constexpr size_t BIG_WIDTH = 16UL;

#define LOG_NUM_THREADS 10
#define NUM_THREADS (1 << LOG_NUM_THREADS)
#define LOG_CONSTRAINTS 16
#define CONSTRAINTS (1 << LOG_CONSTRAINTS)

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

__device__ __forceinline__
size_t bitreverse(size_t n, const size_t l)
{
    return __brevll(n) >> (64ull - l); 
}

template<typename FieldT, int R>
void FFT(cudaStream_t &strm, std::vector<FieldT> &a) {
    cudaStreamCreate(&strm);

    size_t n = (N + R - 1) / R;

    size_t nblocks = (n * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

    if (a.size() != this->m) throw DomainSizeException("step_radix2: expected a.size() == this->m");

    std::vector<FieldT> c(big_m, FieldT::zero());
    std::vector<FieldT> d(big_m, FieldT::zero());

    FieldT omega_i = FieldT::one();
    for (size_t i = 0; i < big_m; ++i)
    {
        c[i] = (i < small_m ? a[i] + a[i+big_m] : a[i]);
        d[i] = omega_i * (i < small_m ? a[i] - a[i+big_m] : a[i]);
        omega_i *= omega;
    }

    std::vector<FieldT> e(small_m, FieldT::zero());
    const size_t compr = 1ul<<(libff::log2(big_m) - libff::log2(small_m));
    for (size_t i = 0; i < small_m; ++i)
    {
        for (size_t j = 0; j < compr; ++j)
        {
            e[i] += d[i + j * small_m];
        }
    }

    _basic_parallel_radix2_FFT<FieldT><<<nblocks, threads_per_block, 0, strm>>>(c, omega.squared());

    bool err = false;
    auto root_of_unity_small_m = libff::get_root_of_unity<FieldT>(small_m, err);
    if (err) {
      throw DomainSizeException("Failed to get_root_of_unity");
    }
    //_basic_radix2_FFT(e, root_of_unity_small_m);
    _basic_parallel_radix2_FFT<FieldT><<<nblocks, threads_per_block, 0, strm>>>(e, root_of_unity_small_m);

    for (size_t i = 0; i < big_m; ++i)
    {
        a[i] = c[i];
    }

    for (size_t i = 0; i < small_m; ++i)
    {
        a[i+big_m] = e[i];
    }
}

template<typename FieldT>
__global__ void
_basic_parallel_radix2_FFT(std::vector<FieldT> &a, const FieldT &omega)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n) {
        const size_t n = a.size(), logn = log2(n);
        if (n != (1u << logn)) throw DomainSizeException("expected n == (1u << logn)");

        /* swapping in place (from Storer's book) */
        for (size_t k = 0; k < n; ++k)
        {
            const size_t rk = libff::bitreverse(k, logn);
            if (k < rk)
                std::swap(a[k], a[rk]);
        }

        size_t m = 1; // invariant: m = 2^{s-1}
        for (size_t s = 1; s <= logn; ++s)
        {
            // w_m is 2^s-th root of unity now
            const FieldT w_m = omega^(n/(2*m));

            asm volatile  ("/* pre-inner */");
            for (size_t k = 0; k < n; k += 2*m)
            {
                FieldT w = FieldT::one();
                for (size_t j = 0; j < m; ++j)
                {
                    const FieldT t = w * a[k+j+m];
                    a[k+j+m] = a[k+j] - t;
                    a[k+j] += t;
                    w *= w_m;
                }
            }
            asm volatile ("/* post-inner */");
            m *= 2;
        }
    }
}

template<typename FieldT>
void _multiply_by_coset(std::vector<FieldT> &a, const FieldT &g)
{
    FieldT u = g;
    for (size_t i = 1; i < a.size(); ++i)
    {
        a[i] *= u;
        u *= g;
    }
}

template<typename FieldT>
void cosetFFT(cudaStream_t &strm, std::vector<FieldT> &a, const FieldT &g)
{
    _multiply_by_coset(a, g);
    FFT(strm, a);
}

template<typename B, typename D, typename V, typename FieldT>
void gpu_cosetFFT(cudaStream_t &strm, D *domain, V &a)
{
    cosetFFT<FieldT>(strm, *a->data, FieldT::multiplicative_generator);
}

template<typename FieldT>  
__global__ void cuda_fft(FieldT *out, FieldT *field) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t log_m = LOG_CONSTRAINTS;
    const size_t length = CONSTRAINTS;
    const size_t block_length = 1ul << (log_m - LOG_NUM_THREADS) ;
    const size_t startidx = idx * block_length;
    assert (CONSTRAINTS == 1ul<<log_m);
    if(startidx > length)
        return;
    FieldT a [block_length];

    //TODO algorithm is non-deterministic because of padding
    FieldT omega_j = FieldT(_mod);
    omega_j = omega_j ^ idx; // pow
    FieldT omega_step = FieldT(_mod);
    omega_step = omega_step ^ (idx << (log_m - LOG_NUM_THREADS));
    
    FieldT elt = FieldT::one();
    //Do not remove log2f(n), otherwise register overflow
    size_t n = block_length, logn = log2f(n);
    assert (n == (1u << logn));
    for (size_t i = 0; i < 1ul<<(log_m - LOG_NUM_THREADS); ++i)
    {
        const size_t ri = bitreverse(i, logn);
        for (size_t s = 0; s < NUM_THREADS; ++s)
        {
            // invariant: elt is omega^(j*idx)
            size_t id = (i + (s<<(log_m - LOG_NUM_THREADS))) % (1u << log_m);
            FieldT tmp = field[id];
            tmp = tmp * elt;
            if (s != 0) tmp = tmp + a[ri];
            a[ri] = tmp;
            elt = elt * omega_step;
        }
        elt = elt * omega_j;
    }

    const FieldT omega_num_cpus = FieldT(_mod) ^ NUM_THREADS;
    size_t m = 1; // invariant: m = 2^{s-1}
    for (size_t s = 1; s <= logn; ++s)
    {
        // w_m is 2^s-th root of unity now
        const FieldT w_m = omega_num_cpus^(n/(2*m));
        for (size_t k = 0; k < n; k += 2*m)
        {
            FieldT w = FieldT::one();
            for (size_t j = 0; j < m; ++j)
            {
                const FieldT t = w;
                w = w * a[k+j+m];
                a[k+j+m] = a[k+j] - t;
                a[k+j] = a[k+j] + t;
                w = w * w_m;
            }
        }
        m = m << 1;
    }
    for (size_t j = 0; j < 1ul<<(log_m - LOG_NUM_THREADS); ++j)
    {
        if(((j << LOG_NUM_THREADS) + idx) < length)
            out[(j<<LOG_NUM_THREADS) + idx] = a[j];
    }
}

template<typename FieldT> 
void best_fft (std::vector<FieldT> &a)
{
	int cnt;
    cudaGetDeviceCount(&cnt);
    printf("CUDA Devices: %d, Field size: %lu, Field count: %lu\n", cnt, sizeof(FieldT), a.size());
    assert(a.size() == CONSTRAINTS);

    size_t blocks = NUM_THREADS / 256 + 1;
    size_t threads = NUM_THREADS > 256 ? 256 : NUM_THREADS;
    printf("NUM_THREADS %u, blocks %lu, threads %lu \n",NUM_THREADS, blocks, threads);

    FieldT *in;
    CUDA_CALL( cudaMalloc((void**)&in, sizeof(FieldT) * a.size()); )
    CUDA_CALL( cudaMemcpy(in, (void**)&a[0], sizeof(FieldT) * a.size(), cudaMemcpyHostToDevice); )

    FieldT *out;
    CUDA_CALL( cudaMalloc(&out, sizeof(FieldT) * a.size()); )
    cuda_fft<FieldT> <<<blocks,threads>>>(out, in);
        
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    CUDA_CALL( cudaMemcpy((void**)&a[0], out, sizeof(FieldT) * a.size(), cudaMemcpyDeviceToHost); )

    CUDA_CALL( cudaDeviceSynchronize();)
}

//List with all templates that should be generated
// template void best_fft(std::vector<fields::Scalar> &v, const fields::Scalar &omg);