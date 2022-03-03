//#include <libfqfft/evaluation_domain/domains/basic_radix2_domain_aux.hpp>
// static constexpr size_t threads_per_block = 256;
// static constexpr size_t BIG_WIDTH = 16UL;

#define LOG_NUM_THREADS 9
#define NUM_THREADS (1 << LOG_NUM_THREADS)
#define LOG_CONSTRAINTS 16
#define CONSTRAINTS (1 << LOG_CONSTRAINTS)

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

#define SIZE (768 / 32)

__device__ u_int32_t _mod [SIZE] = { 610172929, 1586521054, 752685471, 3818738770, 
    2596546032, 1669861489, 1987204260, 1750781161, 3411246648, 3087994277, 
    4061660573, 2971133814, 2707093405, 2580620505, 3902860685, 134068517, 
    1821890675, 1589111033, 1536143341, 3086587728, 4007841197, 270700578, 764593169, 115910};

// __device__ __constant__
// const var MOD_R[16] = {
//     0xd90776e240000001ULL, 0x4ea099170fa13a4fULL,
//     0xd6c381bc3f005797ULL, 0xb9dff97634993aa4ULL,
//     0x3eebca9429212636ULL, 0xb26c5c28c859a99bULL,
//     0x99d124d9a15af79dULL, 0x7fdb925e8a0ed8dULL,
//     0x5eb7e8f96c97d873ULL, 0xb7f997505b8fafedULL,
//     0x10229022eee2cdadULL, 0x1c4c62d92c411ULL

//     , 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL // just to make an even 16
// };

__device__ __forceinline__
size_t bitreverse(size_t n, const size_t l)
{
    return __brevll(n) >> (64ull - l); 
}

// template<typename FieldT>
// __global__ void
// _basic_parallel_radix2_FFT(std::vector<FieldT> &a, const FieldT &omega)
// {
//     int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
//     int elts_per_block = D / BIG_WIDTH;
//     int tileIdx = T / BIG_WIDTH;

//     int idx = elts_per_block * B + tileIdx;

//     if (idx < n) {
//         const size_t n = a.size(), logn = log2(n);
//         if (n != (1u << logn)) throw DomainSizeException("expected n == (1u << logn)");

//         /* swapping in place (from Storer's book) */
//         for (size_t k = 0; k < n; ++k)
//         {
//             const size_t rk = libff::bitreverse(k, logn);
//             if (k < rk)
//                 std::swap(a[k], a[rk]);
//         }

//         size_t m = 1; // invariant: m = 2^{s-1}
//         for (size_t s = 1; s <= logn; ++s)
//         {
//             // w_m is 2^s-th root of unity now
//             const FieldT w_m = omega^(n/(2*m));

//             asm volatile  ("/* pre-inner */");
//             for (size_t k = 0; k < n; k += 2*m)
//             {
//                 FieldT w = FieldT::one();
//                 for (size_t j = 0; j < m; ++j)
//                 {
//                     const FieldT t = w * a[k+j+m];
//                     a[k+j+m] = a[k+j] - t;
//                     a[k+j] += t;
//                     w *= w_m;
//                 }
//             }
//             asm volatile ("/* post-inner */");
//             m *= 2;
//         }
//     }
// }

// template<typename FieldT>
// void _multiply_by_coset(std::vector<FieldT> &a, const FieldT &g)
// {
//     FieldT u = g;
//     for (size_t i = 1; i < a.size(); ++i)
//     {
//         a[i] *= u;
//         u *= g;
//     }
// }

template<typename FieldT>
__global__ void gpu_divide_by_Z_on_coset(var * p, const var *z_inv_on_coset, size_t m)
{
    // const FieldT coset = FieldT::multiplicative_generator;
    //const FieldT Z_inverse_at_coset = compute_vanishing_polynomial(g).inverse();
    // FieldT result;
    // for (size_t i = 0; i < m; ++i)
    // {
    //     FieldT tmp;
    //     FieldT::load(tmp, p + i);
    //     FieldT::mul(tmp, tmp, Z_inverse_at_coset);
    //     // P[i] *= Z_inverse_at_coset;
    //     FieldT::store(p + i, tmp);
    // }
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    // FieldT u = g;
    if (idx < m) {
        FieldT x;
        FieldT y;
        // int x_off = idx * EC::NELTS * ELT_LIMBS;
        int off = idx * ELT_LIMBS;

        FieldT::load(x, p + off);
        FieldT::load(y, z_inv_on_coset + off);

        // We're given W in Monty form for some reason, so undo that.
        FieldT::from_monty(y, y);
        FieldT::mul(x, x, y);

        FieldT::store(p + off, x);
    }
}

template<typename FieldT>
__global__ void
_multiply_by_constant(var *ca, const var *g, size_t n) {
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n) {
        FieldT x;
        FieldT y;

        int off = idx * ELT_LIMBS;

        FieldT::load(x, ca + off);
        FieldT::load(y, g + off);

        FieldT::from_monty(y, y);
        FieldT::mul(x, x, y);

        FieldT::store(ca + off, x);
    }
}



template<typename FieldT>
__global__ void 
_multiply_by_coset(var *ca, const var *g, size_t n)
{
    // FieldT a;
    // FieldT::load(a, ca);
    // FieldT u = g;
    // for (size_t i = 1; i < a.size(); ++i)
    // {
    //     FieldT::mul(a[i], a[i], u);
    //     FieldT::mul(u, u, g);
    // }
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    // FieldT u = g;
    if (idx < n) {
        FieldT a;
        FieldT gen;
        // int x_off = idx * EC::NELTS * ELT_LIMBS;
        int off = idx * ELT_LIMBS;

        FieldT::load(gen, g + off);
        FieldT::load(a, ca + off);

        FieldT::from_monty(gen, gen);
        FieldT::from_monty(a, a);
        // We're given W in Monty form for some reason, so undo that.
        // FieldT::from_monty(w, w);
        FieldT::mul(a, gen, a);
        FieldT::mul(gen, gen, gen);

        FieldT::store(ca + off, a);
    }
}

template<typename FieldT, const size_t BLOCK_SIZE>  
__global__ void cuda_fft(var *out, const var *field, size_t d, const var *omega) {
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    // const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t log_m = log2f(d);
    const size_t length = d;
    // const size_t block_length = 1ul << (log_m - LOG_NUM_THREADS);
    const size_t block_length = BLOCK_SIZE;
    const size_t startidx = idx * block_length;
    assert (d == 1ul<<log_m);
    if(startidx > length)
        return;
    FieldT a [BLOCK_SIZE];
    int s_off = idx * ELT_LIMBS * BLOCK_SIZE;

    for (size_t i = 0; i < 1ul<<(log_m - LOG_NUM_THREADS); ++i) {
        FieldT::load(a[i], field + s_off + i*ELT_LIMBS);
        FieldT::from_monty(a[i], a[i]);
    }
    //TODO algorithm is non-deterministic because of padding
    // FieldT omega_j = FieldT(_mod);
    // omega_j = omega_j ^ idx; // pow
    // FieldT omega_step = FieldT(_mod);
    // omega_step = omega_step ^ (idx << (log_m - LOG_NUM_THREADS));

    FieldT omega_j;
    FieldT::load(omega_j, omega + s_off);
    FieldT::from_monty(omega_j, omega_j);
    FieldT::pow(omega_j, omega_j, idx);
    FieldT omega_step;
    FieldT::load(omega_step, omega + s_off);
    FieldT::from_monty(omega_step, omega_step);
    FieldT::pow(omega_step, omega_step, (idx << (log_m - LOG_NUM_THREADS)));    

    // FieldT elt = FieldT::one();
    FieldT elt;
    FieldT::set_one(elt);
    //Do not remove log2f(n), otherwise register overflow
    size_t n = block_length, logn = log2f(n);
    assert (n == (1u << logn));
    for (size_t i = 0; i < 1ul<<(log_m - LOG_NUM_THREADS); ++i)
    {
        // FieldT::load(a[i], field + s_off + i*ELT_LIMBS);
        // FieldT::from_monty(a[i], a[i]);
        // const size_t ri = bitreverse(i, logn);
        for (size_t s = 0; s < NUM_THREADS; ++s)
        {
            // invariant: elt is omega^(j*idx)
            size_t id = (i + (s<<(log_m - LOG_NUM_THREADS))) % (1u << log_m);
            FieldT tmp = a[id];
            // FieldT::load(tmp, field + (id * ELT_LIMBS));
            // FieldT tmp = field[id];
            // tmp = tmp * elt;
            FieldT::mul(tmp, tmp, elt);
            // if (s != 0) tmp = tmp + a[ri];
            // if (s != 0) FieldT::add(tmp, tmp, a[ri]);
            // a[ri] = tmp;
            //elt = elt * omega_step;
            FieldT::mul(elt, elt, omega_step);
        }
        // elt = elt * omega_j;
        FieldT::mul(elt, elt, omega_j);
    }

    // for (size_t i = 0; i < 1ul<<(log_m - LOG_NUM_THREADS); ++i) {
    //     const size_t ri = bitreverse(i + idx, logn);
    //     if ()
    // }
    for (size_t k = 0; k < n; ++k) {
        const size_t ri = bitreverse(k, logn);
        if (k < ri) {
            FieldT tmp = a[k];
            a[k] = a[ri];
            a[ri] = tmp;
        } 
        // else {
        //     printf("k > ri ");
        // }
    }

    FieldT omega_num;
    FieldT::load(omega_num, omega + s_off);
    FieldT::from_monty(omega_num, omega_num);
    // const FieldT omega_num_cpus = omega_num ^ NUM_THREADS;
    FieldT omega_num_cpus;
    FieldT::pow(omega_num_cpus, omega_num, NUM_THREADS);
    size_t m = 1; // invariant: m = 2^{s-1}
    for (size_t s = 1; s <= logn; ++s)
    {
        // w_m is 2^s-th root of unity now
        // const FieldT w_m = omega_num_cpus^(n/(2*m));
        FieldT w_m;
        FieldT::pow(w_m, omega_num_cpus, n/(2*m));
        for (size_t k = 0; k < n; k += 2*m)
        {
            FieldT w = FieldT::one();
            // FieldT w;
            // FieldT::set_one(w);
            for (size_t j = 0; j < m; ++j)
            {
                const FieldT t = w;
                // w = w * a[k+j+m];
                // a[k+j+m] = a[k+j] - t;
                // a[k+j] = a[k+j] + t;
                // w = w * w_m;
                FieldT::mul(w, w, a[k+j+m]);
                FieldT::sub(a[k+j+m], a[k+j], t);
                FieldT::add(a[k+j], a[k+j], t);
                FieldT::mul(w, w, w_m);
            }
        }
        m = m << 1;
    }

    for (size_t j = 0; j < 1ul<<(log_m - LOG_NUM_THREADS); ++j)
    {
        if(((j << LOG_NUM_THREADS) + idx) < length) {
            // out[(j<<LOG_NUM_THREADS) + idx] = a[j];
            FieldT::store(out + (((j <<LOG_NUM_THREADS) + idx) * ELT_LIMBS), a[j]);
        }
    }
}

template<typename FieldT> 
void best_fft (var *a, size_t m, const var * omega)
{
	int cnt;
    cudaGetDeviceCount(&cnt);
    // printf("CUDA Devices: %d, Field size: %lu, Field count: %lu\n", cnt, sizeof(FieldT), m);
    assert(m == CONSTRAINTS);

    size_t blocks = NUM_THREADS / 256 + 1;
    size_t threads = NUM_THREADS > 256 ? 256 : NUM_THREADS;
    // printf("NUM_THREADS %u, blocks %lu, threads %lu \n",NUM_THREADS, blocks, threads);

    // FieldT *in;
    // CUDA_CALL( cudaMalloc((void**)&in, sizeof(FieldT) * m); )
    // CUDA_CALL( cudaMemcpy(in, (void**)&a[0], sizeof(FieldT) * m, cudaMemcpyHostToDevice); )

    var *out;
    // CUDA_CALL( cudaMalloc(&out, sizeof(FieldT) * m); )
    // cudaMalloc(&out, sizeof(FieldT) * m);
    cudaMalloc(&out, ELT_BYTES * m);

    const size_t log_m = log2f(m);
    const size_t block_length = 1ul << (log_m - LOG_NUM_THREADS);
    const size_t x = 64;
    cuda_fft<FieldT, x><<<blocks,threads>>>(out, a, m, omega);
        
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    a = out;
    // CUDA_CALL( cudaMemcpy((void**)&a[0], out, sizeof(FieldT) * m, cudaMemcpyDeviceToHost); )
    // cudaMemcpy((void**)&a[0], out, sizeof(FieldT) * m, cudaMemcpyDeviceToHost);

    // CUDA_CALL( cudaDeviceSynchronize();)
    cudaDeviceSynchronize();
}


template<typename B, typename FieldT>
void domain_cosetFFT_gpu(var *ca, size_t m, const var * g, const var *omega)
{
    // var gen = 17;
    // FieldT gen;
    // FieldT::load(gen, g);
    // const FieldT g = FieldT::multiplicative_generator;
    size_t blocks = NUM_THREADS / 256 + 1;
    size_t threads = NUM_THREADS > 256 ? 256 : NUM_THREADS;
    // printf("NUM_THREADS %u, blocks %lu, threads %lu \n",NUM_THREADS, blocks, threads);
    _multiply_by_coset<FieldT><<<blocks, threads>>>(ca, g, m);
    cudaDeviceSynchronize();
    // printf("finished multiply by coset\n");
    best_fft<FieldT>(ca, m, omega);
}

template<typename FieldT>
void iFFT(var * a, const var *omega_inv, const var * sconst, size_t m)
{
    best_fft<FieldT>(a, m, omega_inv);
    FieldT x;
        size_t blocks = NUM_THREADS / 256 + 1;
    size_t threads = NUM_THREADS > 256 ? 256 : NUM_THREADS;
    // printf("NUM_THREADS %u, blocks %lu, threads %lu \n",NUM_THREADS, blocks, threads);
    _multiply_by_constant<FieldT><<<blocks, threads>>>(a, sconst, m);
    cudaDeviceSynchronize();
    // FieldT::load(x, a);
    // const FieldT sconst = FieldT(a.size()).inverse();
    // for (size_t i = 0; i < m; ++i)
    // {
    //     *(a + i) *= sconst;
    //     // FieldT::mul(x[i], )
    // }
}

template<typename FieldT>
void icosetFFT(var *a, const var *omega_inv, const var *sconst, const var *g_inv, size_t m)
{
    iFFT<FieldT>(a, omega_inv, sconst, m);
    size_t blocks = NUM_THREADS / 256 + 1;
    size_t threads = NUM_THREADS > 256 ? 256 : NUM_THREADS;
    // printf("NUM_THREADS %u, blocks %lu, threads %lu \n",NUM_THREADS, blocks, threads);
    _multiply_by_coset<FieldT><<<blocks, threads>>>(a, g_inv, m);
    cudaDeviceSynchronize();
}

// template<typename FieldT>
// void domain_cosetFFT_gpu(std::vector<FieldT> &a, const FieldT &g, size_t d)
// {
//     // const FieldT g = FieldT::multiplicative_generator;
//     // _multiply_by_coset<FieldT>(*a->data, g);
//     // best_fft<FieldT>(*a->data, d);

//     _multiply_by_coset<FieldT>(a, g);
//     best_fft<FieldT>(a, d);
// }

// Using operators
// template<typename FieldT>  
// __global__ void cuda_fft(FieldT *out, FieldT *field, size_t d, const var *omega) {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const size_t log_m = d;
//     const size_t length = 1 << d;
//     const size_t block_length = 1ul << (log_m - LOG_NUM_THREADS) ;
//     const size_t startidx = idx * block_length;
//     assert (d == 1ul<<log_m);
//     if(startidx > length)
//         return;
//     FieldT a [block_length];

//     //TODO algorithm is non-deterministic because of padding
//     FieldT omega_j;
//     FieldT::load(omega_j, omega);
//     // FieldT omega_j = FieldT(_mod);
//     omega_j = omega_j ^ idx; // pow
//     // FieldT omega_step = FieldT();
//     FieldT omega_step;
//     FieldT::load(omega_step, omega);
//     omega_step = omega_step ^ (idx << (log_m - LOG_NUM_THREADS));
    

//     // FieldT elt = FieldT::one();
//     FieldT elt;
//     FieldT::set_one(elt);
//     //Do not remove log2f(n), otherwise register overflow
//     size_t n = block_length, logn = log2f(n);
//     assert (n == (1u << logn));
//     for (size_t i = 0; i < 1ul<<(log_m - LOG_NUM_THREADS); ++i)
//     {
//         const size_t ri = bitreverse(i, logn);
//         for (size_t s = 0; s < NUM_THREADS; ++s)
//         {
//             // invariant: elt is omega^(j*idx)
//             size_t id = (i + (s<<(log_m - LOG_NUM_THREADS))) % (1u << log_m);
//             FieldT tmp = field[id];
//             tmp = tmp * elt;
//             if (s != 0) tmp = tmp + a[ri];
//             a[ri] = tmp;
//             elt = elt * omega_step;
//         }
//         elt = elt * omega_j;
//     }

//     FieldT omega_num;
//     FieldT::load(omega_num, omega);
//     const FieldT omega_num_cpus = omega_num ^ NUM_THREADS;
//     size_t m = 1; // invariant: m = 2^{s-1}
//     for (size_t s = 1; s <= logn; ++s)
//     {
//         // w_m is 2^s-th root of unity now
//         const FieldT w_m = omega_num_cpus^(n/(2*m));
//         for (size_t k = 0; k < n; k += 2*m)
//         {
//             FieldT w = FieldT::one();
//             // FieldT w;
//             // FieldT::set_one(w);
//             for (size_t j = 0; j < m; ++j)
//             {
//                 const FieldT t = w;
//                 w = w * a[k+j+m];
//                 a[k+j+m] = a[k+j] - t;
//                 a[k+j] = a[k+j] + t;
//                 w = w * w_m;
//             }
//         }
//         m = m << 1;
//     }
//     for (size_t j = 0; j < 1ul<<(log_m - LOG_NUM_THREADS); ++j)
//     {
//         if(((j << LOG_NUM_THREADS) + idx) < length)
//             out[(j<<LOG_NUM_THREADS) + idx] = a[j];
//     }
// }

// template<typename FieldT> 
// void best_fft (const var *a, size_t m, const var * omega)
// {
// 	int cnt;
//     cudaGetDeviceCount(&cnt);
//     printf("CUDA Devices: %d, Field size: %lu, Field count: %lu\n", cnt, sizeof(FieldT), m);
//     assert(m == CONSTRAINTS);

//     size_t blocks = NUM_THREADS / 256 + 1;
//     size_t threads = NUM_THREADS > 256 ? 256 : NUM_THREADS;
//     printf("NUM_THREADS %u, blocks %lu, threads %lu \n",NUM_THREADS, blocks, threads);

//     FieldT *in;
//     CUDA_CALL( cudaMalloc((void**)&in, sizeof(FieldT) * m); )
//     CUDA_CALL( cudaMemcpy(in, (void**)&a[0], sizeof(FieldT) * m, cudaMemcpyHostToDevice); )

//     FieldT *out;
//     CUDA_CALL( cudaMalloc(&out, sizeof(FieldT) * m); )
    
//     // const size_t log_m = log2f(m);
//     // const size_t block_length = 1ul << (log_m - LOG_NUM_THREADS);
//     const size_t x = 32;
//     cuda_fft<FieldT, x> <<<blocks,threads>>>(out, in, m, omega);
        
//     cudaError_t error = cudaGetLastError();
//     if(error != cudaSuccess)
//     {
//         printf("CUDA error: %s\n", cudaGetErrorString(error));
//         exit(-1);
//     }

//     CUDA_CALL( cudaMemcpy((void**)&a[0], out, sizeof(FieldT) * m, cudaMemcpyDeviceToHost); )

//     CUDA_CALL( cudaDeviceSynchronize();)
// }

//List with all templates that should be generated
// template void best_fft(std::vector<fields::Scalar> &v, const fields::Scalar &omg);