#include <cstdint>
#include <vector>
#include <chrono>
#include <memory>
#include <cooperative_groups.h>

#include "curves.cu"

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

// template <typename EC, typename FieldT>
// __inline__ __device__
// EC warpReduceSum(EC x) {
//     EC z, y;
//     #pragma unroll
//     for (int offset = warpSize/2; offset > 0; offset /= 2) {
//         EC y = FieldT::shuffle_down(__activemask(), x, offset);
//         EC::add(z, x, y)
//         //x = x + y;
//     }
//     return z;
// }

// template <typename FieldT>
// extern __shared__ FieldT sMem[];

// template <typename EC, typename FieldT>
// __inline__ __device__
// FieldT blockReduceSum(EC x) {
//     int lane = threadIdx.x % warpSize;
//     int warpId = threadIdx.x / warpSize;
//     x = warpReduceSum<EC>(x); 
//     if (lane==0) sMem<EC>[warpId]=x;
//     __syncthreads();
//     if(threadIdx.x < blockDim.x / warpSize)
//         x = sMem<EC>[lane];
//     else {
//         //x = FieldT::zero();
//         EC::set_zero(x)
//     }
//     if (warpId==0) x = warpReduceSum<EC>(x);
//     return x;
// }

// template <typename EC, typename FieldT>
// __global__ void
// deviceReduceKernelSecond(FieldT *out, const FieldT *resIn, const size_t n) { 
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     EC sum;
//     EC::set_zero(sum);
//     if (idx < n) {
//         EC z;
//         for(int i = idx; i < n; i += (blockDim.x * gridDim.x)){
//             //sum = sum + resIn[i];
//             EC::add(z, sum, resIn[i])
//         }
//         sum = blockReduceSum<FieldT>(sum);
//     }   
//     if (threadIdx.x==0) // Store the end result
//         out[blockIdx.x] = sum; 
// }

// template <typename EC, typename FieldT, typename FieldMul>
// __global__ void 
// deviceReduceKernel(FieldT *result, const FieldT *a, const FieldMul *mul, const size_t n) {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     EC x;
//     EC::set_zero(x);
//     if (idx < n) {
//         for(int i = idx; i < n; i += (blockDim.x * gridDim.x)){
//             const FieldT tmp = a[i] * mul[i];
//             sum = sum + tmp;   
//         }
//         sum = blockReduceSum<FieldT>(sum);
//     }
//     if (threadIdx.x==0)
//         result[blockIdx.x] = sum;
// }

// // Multiexp is a function that performs a multiplication and a summation of all elements.
// template<typename EC> 
// void multiexp (var *out, const var *multiples, const var *scalars, size_t N) {
//     //assert(a.size() == mul.size());
//     typedef typename EC::group_type FieldT;
//     size_t threads = sizeof(FieldT) == 96 ? 256 : 128;
//     size_t blocks = min((a.size() + threads - 1) / threads, (unsigned long)256);
//     size_t sMem = threads * sizeof(FieldT);
    
//     FieldT *in;
//     CUDA_CALL( cudaMalloc((void**)&in, sizeof(FieldT) * a.size()); )
//     CUDA_CALL( cudaMemcpy(in, (void**)&a[0], sizeof(FieldT) * a.size(), cudaMemcpyHostToDevice); )

//     FieldMul *cmul;
//     CUDA_CALL( cudaMalloc((void**)&cmul, sizeof(FieldMul) * a.size()); )
//     CUDA_CALL( cudaMemcpy(cmul, (void**)&mul[0], sizeof(FieldMul) * a.size(), cudaMemcpyHostToDevice); )

//     FieldT *temp;
//     CUDA_CALL( cudaMalloc((void**)&temp, sizeof(FieldT) * blocks); )
//     deviceReduceKernel<EC,FieldT,FieldT><<<blocks, threads, sMem>>>(temp, in, cmul, a.size());
    
//     FieldT *last;
//     CUDA_CALL( cudaMalloc(&out, sizeof(FieldT)); )
//     deviceReduceKernelSecond<EC,FieldT><<<1, 256, sMem>>>(last, temp, blocks);

//     cudaError_t error = cudaGetLastError();
//     if(error != cudaSuccess)
//     {
//         printf("CUDA error: %s\n", cudaGetErrorString(error));
//         exit(-2);
//     }

//     //FieldT cpuResult;
//     cudaMemcpy((void**)&out, last, sizeof(FieldT), cudaMemcpyDeviceToHost);
//     //return cpuResult;
// }

// C is the size of the precomputation
// R is the number of points we're handling per thread
template< typename EC, int C = 4, int RR = 8 >
__global__ void
ec_multiexp_straus(var *out, const var *multiples_, const var *scalars_, size_t N)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    size_t n = (N + RR - 1) / RR;
    if (idx < n) {
        // TODO: Treat remainder separately so R can remain a compile time constant
        size_t R = (idx < n - 1) ? RR : (N % RR);

        typedef typename EC::group_type Fr;
        static constexpr int JAC_POINT_LIMBS = 3 * EC::field_type::DEGREE * ELT_LIMBS;
        static constexpr int AFF_POINT_LIMBS = 2 * EC::field_type::DEGREE * ELT_LIMBS;
        int out_off = idx * JAC_POINT_LIMBS;
        int m_off = idx * RR * AFF_POINT_LIMBS;
        int s_off = idx * RR * ELT_LIMBS;

        Fr scalars[RR];
        for (int j = 0; j < R; ++j) {
            Fr::load(scalars[j], scalars_ + s_off + j*ELT_LIMBS);
            Fr::from_monty(scalars[j], scalars[j]);
        }

        const var *multiples = multiples_ + m_off;
        // TODO: Consider loading multiples and/or scalars into shared memory

        // i is smallest multiple of C such that i > 753
        int i = C * ((753 + C - 1) / C); // C * ceiling(753/C)
        assert((i - C * 753) < C);
        static constexpr var C_MASK = (1U << C) - 1U;

        EC x;
        EC::set_zero(x);
        while (i >= C) {
            EC::mul_2exp<C>(x, x);
            i -= C;

            int q = i / digit::BITS, r = i % digit::BITS;
            for (int j = 0; j < R; ++j) {
                //(scalars[j][q] >> r) & C_MASK
                auto g = fixnum::layout();
                var s = g.shfl(scalars[j].a, q);
                var win = (s >> r) & C_MASK;
                // Handle case where C doesn't divide digit::BITS
                int bottom_bits = digit::BITS - r;
                // detect when window overlaps digit boundary
                if (bottom_bits < C) {
                    s = g.shfl(scalars[j].a, q + 1);
                    win |= (s << bottom_bits) & C_MASK;
                }
                if (win > 0) {
                    EC m;
                    //EC::add(x, x, multiples[win - 1][j]);
                    EC::load_affine(m, multiples + ((win-1)*N + j)*AFF_POINT_LIMBS);
                    EC::mixed_add(x, x, m);
                }
            }
        }
        EC::store_jac(out + out_off, x);
    }
}

template< typename EC >
__global__ void
ec_multiexp(var *X, const var *W, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n) {
        typedef typename EC::group_type Fr;
        EC x;
        Fr w;
        int x_off = idx * EC::NELTS * ELT_LIMBS;
        int w_off = idx * ELT_LIMBS;

        EC::load_affine(x, X + x_off);
        Fr::load(w, W + w_off);

        // We're given W in Monty form for some reason, so undo that.
        Fr::from_monty(w, w);
        EC::mul(x, w.a, x);

        EC::store_jac(X + x_off, x);
    }
}

template< typename EC >
__global__ void
ec_sum_all(var *X, const var *Y, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n) {
        EC z, x, y;
        int off = idx * EC::NELTS * ELT_LIMBS;

        EC::load_jac(x, X + off);
        EC::load_jac(y, Y + off);

        EC::add(z, x, y);

        EC::store_jac(X + off, z);
    }
}

static constexpr size_t threads_per_block = 256;

template< typename EC, int C, int R >
void
ec_reduce_straus(cudaStream_t &strm, var *out, const var *multiples, const var *scalars, size_t N)
{
    cudaStreamCreate(&strm);

    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;
    size_t n = (N + R - 1) / R;

    size_t nblocks = (n * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

    ec_multiexp_straus<EC, C, R><<< nblocks, threads_per_block, 0, strm>>>(out, multiples, scalars, N);

    size_t r = n & 1, m = n / 2;
    for ( ; m != 0; r = m & 1, m >>= 1) {
        nblocks = (m * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

        ec_sum_all<EC><<<nblocks, threads_per_block, 0, strm>>>(out, out + m*pt_limbs, m);
        if (r)
            ec_sum_all<EC><<<1, threads_per_block, 0, strm>>>(out, out + 2*m*pt_limbs, 1);
    }
}

template< typename EC >
void
ec_reduce(cudaStream_t &strm, var *X, const var *w, size_t n)
{
    cudaStreamCreate(&strm);

    size_t nblocks = (n * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

    // FIXME: Only works on Pascal and later.
    //auto grid = cg::this_grid();
    ec_multiexp<EC><<< nblocks, threads_per_block, 0, strm>>>(X, w, n);

    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;

    size_t r = n & 1, m = n / 2;
    for ( ; m != 0; r = m & 1, m >>= 1) {
        nblocks = (m * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

        ec_sum_all<EC><<<nblocks, threads_per_block, 0, strm>>>(X, X + m*pt_limbs, m);
        if (r)
            ec_sum_all<EC><<<1, threads_per_block, 0, strm>>>(X, X + 2*m*pt_limbs, 1);
        // TODO: Not sure this is really necessary.
        //grid.sync();
    }
}

static inline double as_mebibytes(size_t n) {
    return n / (long double)(1UL << 20);
}

void print_meminfo(size_t allocated) {
    size_t free_mem, dev_mem;
    cudaMemGetInfo(&free_mem, &dev_mem);
    fprintf(stderr, "Allocated %zu bytes; device has %.1f MiB free (%.1f%%).\n",
            allocated,
            as_mebibytes(free_mem),
            100.0 * free_mem / dev_mem);
}

struct VarWithStream {  
    cudaStream_t stream;
    var *mem;
};
struct CudaFree {
    void operator()(var *mem) { cudaFree(mem); }
};
struct CudaFreeAsync {
    void operator()(VarWithStream *var_with_stream) { cudaFreeAsync(var_with_stream->mem, var_with_stream->stream); }
};
typedef std::unique_ptr<var, CudaFree> var_ptr;
typedef std::unique_ptr<VarWithStream, CudaFreeAsync> var_ptr_async;

var_ptr
allocate_memory_managed(size_t nbytes, int dbg = 0) {
    var *mem = nullptr;
    cudaMallocManaged(&mem, nbytes);
    if (mem == nullptr) {
        fprintf(stderr, "Failed to allocate enough device memory\n");
        abort();
    }
    if (dbg)
        print_meminfo(nbytes);
    return var_ptr(mem);
}

var_ptr
allocate_memory(size_t nbytes, int dbg = 0) {
    var *mem = nullptr;
    cudaMalloc(&mem, nbytes);
    if (mem == nullptr) {
        fprintf(stderr, "Failed to allocate enough device memory\n");
        abort();
    }
    if (dbg)
        print_meminfo(nbytes);
    return var_ptr(mem);
}

var_ptr_async
allocate_memory_async(size_t nbytes, cudaStream_t &strm, int dbg = 0) {
    struct VarWithStream *var_async;
    // var *mem = nullptr;
    var_async->mem = nullptr;
    var *mem = var_async->mem;
    cudaMallocAsync(&mem, nbytes, strm);
    if (mem == nullptr) {
        fprintf(stderr, "Failed to allocate enough device memory\n");
        abort();
    }
    if (dbg)
        print_meminfo(nbytes);
    // var_async->mem = mem;
    var_async->stream = strm;
    return var_ptr_async(var_async);
}

var_ptr
load_scalars(size_t n, FILE *inputs)
{
    static constexpr size_t scalar_bytes = ELT_BYTES;
    size_t total_bytes = n * scalar_bytes;
    printf("total scalar bytes: %zu\n", total_bytes);
    auto mem = allocate_memory(total_bytes, 1);

    void *scalars_buffer = (void *) malloc (total_bytes);
    if (fread(scalars_buffer, total_bytes, 1, inputs) < 1) {
        fprintf(stderr, "Failed to read scalars\n");
        abort();
    }
    cudaMemcpy(mem.get(), scalars_buffer, total_bytes, cudaMemcpyHostToDevice); 
    free(scalars_buffer);
    return mem;
}

var_ptr
load_scalars_async(cudaStream_t &strm, size_t n, FILE *inputs)
{
    //cudaStreamCreate(&strm);
    static constexpr size_t scalar_bytes = ELT_BYTES;
    size_t total_bytes = n * scalar_bytes;
    printf("total scalar bytes: %zu\n", total_bytes);
    // auto mem = allocate_memory_async(strm, total_bytes, 1);
    auto mem = allocate_memory(total_bytes, 1);

    void *scalars_buffer = (void *) malloc (total_bytes);
    if (fread(scalars_buffer, total_bytes, 1, inputs) < 1) {
        fprintf(stderr, "Failed to read scalars\n");
        abort();
    }
    cudaMemcpyAsync(mem.get(), scalars_buffer, total_bytes, cudaMemcpyHostToDevice, strm); 
    free(scalars_buffer);
    return mem;
}

template< typename EC >
var_ptr
load_points(size_t n, FILE *inputs)
{
    typedef typename EC::field_type FF;

    static constexpr size_t coord_bytes = FF::DEGREE * ELT_BYTES;
    static constexpr size_t aff_pt_bytes = 2 * coord_bytes;
    static constexpr size_t jac_pt_bytes = 3 * coord_bytes;

    size_t total_aff_bytes = n * aff_pt_bytes;
    size_t total_jac_bytes = n * jac_pt_bytes;

    auto mem = allocate_memory(total_jac_bytes);
    if (fread((void *)mem.get(), total_aff_bytes, 1, inputs) < 1) {
        fprintf(stderr, "Failed to read all curve poinst\n");
        abort();
    }

    // insert space for z-coordinates
    char *cmem = reinterpret_cast<char *>(mem.get()); //lazy
    for (size_t i = n - 1; i > 0; --i) {
        char tmp_pt[aff_pt_bytes];
        memcpy(tmp_pt, cmem + i * aff_pt_bytes, aff_pt_bytes);
        memcpy(cmem + i * jac_pt_bytes, tmp_pt, aff_pt_bytes);
    }
    return mem;
}

template< typename EC >
var_ptr
load_points_affine(size_t n, FILE *inputs)
{
    typedef typename EC::field_type FF;

    static constexpr size_t coord_bytes = FF::DEGREE * ELT_BYTES;
    static constexpr size_t aff_pt_bytes = 2 * coord_bytes;

    size_t total_aff_bytes = n * aff_pt_bytes;
    printf("total affine bytes: %zu\n", total_aff_bytes);
    auto mem = allocate_memory(total_aff_bytes, 1);

    void *aff_bytes_buffer = (void *) malloc (total_aff_bytes);
    if (fread(aff_bytes_buffer, total_aff_bytes, 1, inputs) < 1) {
        fprintf(stderr, "Failed to read all curve poinst\n");
        abort();
    }
    //printf("aff_bytes_buffer: %p\n", aff_bytes_buffer);
    cudaMemcpy(mem.get(), aff_bytes_buffer, total_aff_bytes, cudaMemcpyHostToDevice); 
    //printf("mem in load_points_affine after cudaMemcpy: %zu\n", mem.get());
    free(aff_bytes_buffer);
    return mem;
}

template< typename EC >
var_ptr
load_points_affine_async(cudaStream_t &strm, size_t n, FILE *inputs)
{
    typedef typename EC::field_type FF;

    static constexpr size_t coord_bytes = FF::DEGREE * ELT_BYTES;
    static constexpr size_t aff_pt_bytes = 2 * coord_bytes;

    size_t total_aff_bytes = n * aff_pt_bytes;
    printf("total affine bytes: %zu\n", total_aff_bytes);

    // void *aff_bytes_buffer = (void *) malloc (total_aff_bytes);
    void *aff_bytes_buffer;
    cudaMallocHost((void **)&aff_bytes_buffer, total_aff_bytes);
    if (fread(aff_bytes_buffer, total_aff_bytes, 1, inputs) < 1) {
        fprintf(stderr, "Failed to read all curve poinst\n");
        abort();
    }
    cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
    // auto mem = allocate_memory_async(strm, total_aff_bytes, 1);
    auto mem = allocate_memory(total_aff_bytes, 1);

    //printf("aff_bytes_buffer: %p\n", aff_bytes_buffer);
    cudaMemcpyAsync(mem.get(), aff_bytes_buffer, total_aff_bytes, cudaMemcpyHostToDevice, strm); 
    //printf("mem in load_points_affine after cudaMemcpy: %zu\n", mem.get());
    // cudaFreeHost(aff_bytes_buffer);
    // free(aff_bytes_buffer);
    return mem;
} 