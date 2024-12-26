#include "ex2.h"
#include <cuda/atomic>

#define AVAILBLE_STREAM (-1)

#define INVALID_IMAGE (-1)
#define STOPPED_IMAGE (-2)

#define GREY_LEVELS 256
#define HISTOGRAM_SIZE 256
#define THREADS_COUNT 1024
#define TILE_SIZE (TILE_WIDTH * TILE_WIDTH)
#define TILE_PER_IMAGE (TILE_COUNT * TILE_COUNT)
#define MAP_SIZE (TILE_COUNT * TILE_COUNT * GREY_LEVELS)

__device__ void prefix_sum(int arr[], int arr_size) {
	const int tid = threadIdx.x; 
	int increment;

	for (int stride = 1 ; stride < arr_size ; stride *= 2)
	{
		if (tid >= stride && tid < arr_size)
		{
			increment = arr[tid - stride];
		}
		__syncthreads();
		if (tid >= stride && tid < arr_size)
		{
			arr[tid] += increment;
		}
		__syncthreads();
	}
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__device__
void get_histogram(int* hist, uchar* all_in, int tile_row, int tile_col)
{
	const int tid = threadIdx.x;
	const int thread_work = (TILE_WIDTH * TILE_WIDTH) / blockDim.x;
	const int threads_per_row = TILE_WIDTH / thread_work;
	const int x_index = (TILE_WIDTH * tile_row) + (tid / threads_per_row);
	const int y_index = (TILE_WIDTH * tile_col) + ((tid % threads_per_row) * thread_work);
	int color_value = 0;
	int index = 0;
	
	for(int j = 0 ; j < thread_work ; j++)
	{
		index = x_index * IMG_WIDTH + y_index + j;
		color_value = all_in[index];
		atomicAdd(&hist[color_value], 1);
	}  
}

__device__
void get_maps(int* cdf, uchar* maps, int tile_row, int tile_col)
{
	const int tid = threadIdx.x;
	if (tid >= GREY_LEVELS)
	{
		return;
	}

	const int tile_size = TILE_WIDTH * TILE_WIDTH;
	const int maps_start_index = ((tile_row * TILE_COUNT) + tile_col) * GREY_LEVELS;

	maps[maps_start_index + tid] = (float(cdf[tid]) * (GREY_LEVELS - 1)) / (tile_size);
}

__device__
void process_image(uchar *in, uchar *out, uchar* maps) 
{
	__shared__ int hist[GREY_LEVELS];

	for (int tile_row = 0 ; tile_row < TILE_COUNT ; tile_row++)
	{
		for (int tile_col = 0 ; tile_col < TILE_COUNT ; tile_col++)
		{
			memset(hist, 0, GREY_LEVELS * sizeof(int));
			__syncthreads();

			get_histogram(hist, in, tile_row, tile_col);
			__syncthreads();          
	
			prefix_sum(hist, GREY_LEVELS); 
			__syncthreads();            
		
			get_maps(hist, maps, tile_row, tile_col); 
			__syncthreads();    
		}
	}
	
	__syncthreads();
	interpolate_device(maps, in, out);

	return; 
}

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
	process_image(in, out, maps);
}


/* Helpfull Struct */
typedef struct 
{
	cudaStream_t stream;
	int streamImageId;
	uchar *taskMaps;
} Stream_Wrap;

class streams_server : public image_processing_server
{
private:
	// TODO define stream server context (memory buffers, streams, etc...)
	Stream_Wrap streams[STREAM_COUNT];
	uchar* stream_to_imgin[STREAM_COUNT];
	uchar* stream_to_imgout[STREAM_COUNT];

	

public:
	streams_server()
	{
		// TODO initialize context (memory buffers, streams, etc...)
		for (int i = 0; i < STREAM_COUNT; i++) {
			CUDA_CHECK(cudaStreamCreate(&streams[i].stream));
			streams[i].streamImageId = AVAILBLE_STREAM; // avialble
			CUDA_CHECK(cudaMalloc((void**)&(streams[i].taskMaps), TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE));
			CUDA_CHECK(cudaMalloc(&stream_to_imgin[i], IMG_WIDTH * IMG_HEIGHT));
			CUDA_CHECK(cudaMalloc(&stream_to_imgout[i], IMG_WIDTH * IMG_HEIGHT));
		}
	
	}

	~streams_server() override
	{
		// TODO free resources allocated in constructor
		for (int i = 0; i < STREAM_COUNT; i++) {
			CUDA_CHECK(cudaStreamDestroy(streams[i].stream));
			CUDA_CHECK(cudaFree(streams[i].taskMaps));
			CUDA_CHECK(cudaFree(stream_to_imgin[i]));
			CUDA_CHECK(cudaFree(stream_to_imgout[i]));
		}
	}

	bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
	{
		for (int i = 0; i < STREAM_COUNT; i++)
		{
			if (streams[i].streamImageId == AVAILBLE_STREAM)
			{
				streams[i].streamImageId = img_id;
				
				CUDA_CHECK(cudaMemcpyAsync(stream_to_imgin[i], img_in, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice, streams[i].stream));
				
				process_image_kernel<<<1, 1024, 0, streams[i].stream>>>(stream_to_imgin[i], stream_to_imgout[i], streams[i].taskMaps);
				
				CUDA_CHECK(cudaMemcpyAsync(img_out, stream_to_imgout[i], IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost, streams[i].stream));

				return true;
			}
		}
		return false;
	}

	bool dequeue(int *img_id) override
	{
		// TODO query (don't block) streams for any completed requests.
		for (int i = 0; i < STREAM_COUNT; i++)
		{
			if (streams[i].streamImageId != AVAILBLE_STREAM)
			{
				cudaError_t status = cudaStreamQuery(streams[i].stream); // TODO query diffrent stream each iteration
				switch (status) {
				case cudaSuccess:
					// TODO return the img_id of the request that was completed.
					*img_id = streams[i].streamImageId;
					streams[i].streamImageId = AVAILBLE_STREAM;
					return true;
				case cudaErrorNotReady:
			continue;
				default:
					CUDA_CHECK(status);
					return false;
				}
			}
			
		}

		return false;
	}
};

std::unique_ptr<image_processing_server> create_streams_server()
{
	return std::make_unique<streams_server>();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Request
{
		int imgID;	
		uchar *imgIn;
		uchar *imgOut;
};

// TODO implement a lock
class TTASLock 
{
private:
	cuda::atomic<int, cuda::thread_scope_device> _lock;

public:
	__device__ TTASLock() :
		_lock(0)
	{}

	__device__ void lock() 
	{
		// Entry protocol as specified
		do {
			// Spin using normal instructions until the lock is free
			while (_lock.load(cuda::memory_order_relaxed) == 1) {
				; //halt
			}
		} while (_lock.exchange(1, cuda::memory_order_acquire));  // Attempt to acquire the lock
		cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
	}
 
	__device__ void unlock() 
	{
		_lock.store(0, cuda::memory_order_release);
	}
};

__device__ TTASLock* gpu_pop_lock;
__device__ TTASLock* gpu_push_lock;


__global__ void alloc_gpu_lock()
{
	const uint32_t tid = threadIdx.x;
	const uint32_t bid = blockIdx.x;

	// allocate and initialize the gpu lock
	if (tid == 0 && bid == 0)
	{
		gpu_pop_lock = new TTASLock();
		gpu_push_lock = new TTASLock();
	}
}

__global__ void free_gpu_lock()
{
	const uint32_t tid = threadIdx.x;
	const uint32_t bid = blockIdx.x;

	// free the gpu lock
	if (tid == 0 && bid == 0)
	{
		delete gpu_push_lock;
		delete gpu_pop_lock;
	}
}

// TODO implement a MPMC queue
template <typename T> 
class RingBuffer 
{
private:
	int N;
	cuda::atomic<int> _head, _tail;
	T* _mailbox;

public:
	RingBuffer() = default;
	explicit RingBuffer(int n):
		N(n),
		_head(0),
		_tail(0)
	{  
		CUDA_CHECK(cudaMallocHost(&_mailbox, sizeof(T) * N));
	}
	~RingBuffer()
	{
		CUDA_CHECK(cudaFreeHost(_mailbox));
	}

	__device__ __host__ bool push(const T &data) 
	{
		int tail = _tail.load(cuda::memory_order_relaxed);
		int head = _head.load(cuda::memory_order_acquire);

		//printf("Push: tail = %d, head = %d\n", tail, head);
	
		if ((tail - head) % (2 * N) >= N - 1 )
		{
			//printf("Push failed: Queue is full. tail = %d, head = %d\n", tail, head);
			return false;
		}
	
		_mailbox[_tail % N] = data;
		_tail.store(tail + 1, cuda::memory_order_release);
	
		//printf("Push successful: new tail = %d, head = %d\n", tail + 1, head);
	
		return true;
	}
	
	__device__ __host__ T pop() 
	{
		int head = _head.load(cuda::memory_order_relaxed);
		int tail = _tail.load(cuda::memory_order_acquire);

		T item;
	
		//printf("Pop: head = %d, tail = %d\n", head, tail);
	
		if ((tail - head) % (2 * N) <= 0)
		{
			//printf("Pop failed: Queue is empty. tail = %d, head = %d\n", tail, head);
			item.imgID = INVALID_IMAGE;
			return item;
		}
	
		item = _mailbox[_head % N];
		_head.store(head + 1, cuda::memory_order_release);
	
		//printf("Pop successful: new head = %d, tail = %d\n", head + 1, tail);
	
		return item;
	}

};

// TODO implement the persistent kernel

__global__ void persistent_gpu_kernel(RingBuffer<Request>* cpu_to_gpu_queue, RingBuffer<Request>* gpu_to_cpu_queue, uchar* maps)
{
	const uint32_t tid = threadIdx.x;
	const uint32_t bid = blockIdx.x;

	uchar* tb_map = maps + bid * TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE;
	__shared__ Request req;

	while(true)
	{
		if (tid == 0)
		{
			gpu_pop_lock->lock();
			req = cpu_to_gpu_queue->pop(); 
			gpu_pop_lock->unlock();
		}
		__syncthreads();

		// Halt all threads within this thread block
		if (req.imgID == STOPPED_IMAGE)
		{
			return; 
		}

		if (req.imgID != INVALID_IMAGE)
		{    

			process_image(req.imgIn, req.imgOut, tb_map);
			__syncthreads();

			if (tid == 0)
			{
				gpu_push_lock->lock();
				while(!gpu_to_cpu_queue->push(req))
				{
					; //halt
				}
				gpu_push_lock->unlock();
			}
		}
	}
}

// TODO implement a function for calculating the threadblocks count

int calc_thread_blocks(int threads)
{
	int cuda_device_num;
	CUDA_CHECK(cudaGetDevice(&cuda_device_num));

	cudaDeviceProp Prop;
	CUDA_CHECK(cudaGetDeviceProperties(&Prop, cuda_device_num));

	struct cudaFuncAttributes func;
	CUDA_CHECK(cudaFuncGetAttributes(&func,persistent_gpu_kernel));

	//constraints
	int max_shared_mem_sm = Prop.sharedMemPerMultiprocessor;
	int max_regs_per_sm = Prop.regsPerMultiprocessor;
	int max_threads_per_sm = Prop.maxThreadsPerMultiProcessor;

	int reg_per_thread = 32; // in make file
	const uint32_t used_smem_per_block = func.sharedSizeBytes;

	int max_tb_mem_constraint = max_shared_mem_sm / used_smem_per_block;
	int max_tb_reg_constraint = max_regs_per_sm / (reg_per_thread * threads);
	int max_tb_threads_constraint = max_threads_per_sm / threads;

	int max_tb = std::min(max_tb_mem_constraint,std::min(max_tb_reg_constraint, max_tb_threads_constraint));
	int max_num_sm = Prop.multiProcessorCount;
	return max_num_sm * max_tb;

}



class queue_server : public image_processing_server
{
private:
	RingBuffer<Request>* cpu_to_gpu_queue;
	RingBuffer<Request>* gpu_to_cpu_queue;
	uchar* maps;
	uint32_t blocks_count;
public:
	queue_server(int threads)
	{
		// TODO initialize host state
		blocks_count = calc_thread_blocks(threads);   
		//printf("Using device %d\n", blocks_count); 
		int queue_size = std::pow(2, std::ceil(std::log(16*blocks_count)/std::log(2)));

		CUDA_CHECK(cudaMalloc(&maps, blocks_count * MAP_SIZE));
		CUDA_CHECK(cudaMallocHost(&cpu_to_gpu_queue, sizeof(RingBuffer<Request>)));
		CUDA_CHECK(cudaMallocHost(&gpu_to_cpu_queue, sizeof(RingBuffer<Request>)));
		new(cpu_to_gpu_queue) RingBuffer<Request>(queue_size);
		new(gpu_to_cpu_queue) RingBuffer<Request>(queue_size);
		// create gpu lock
		alloc_gpu_lock<<<1, 1>>>();
		CUDA_CHECK(cudaDeviceSynchronize());
		// TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
		persistent_gpu_kernel<<<blocks_count, threads>>>(cpu_to_gpu_queue, gpu_to_cpu_queue, maps);
	}

	~queue_server() override
	{
		for (uint32_t i = 0 ; i < blocks_count; i++)
		{
			enqueue(STOPPED_IMAGE, nullptr, nullptr);
		}
		CUDA_CHECK(cudaDeviceSynchronize()); 
		// TODO free resources allocated in constructor
		cpu_to_gpu_queue->~RingBuffer<Request>();
		gpu_to_cpu_queue->~RingBuffer<Request>();
		CUDA_CHECK(cudaFree(maps));
		CUDA_CHECK(cudaFreeHost(cpu_to_gpu_queue));
		CUDA_CHECK(cudaFreeHost(gpu_to_cpu_queue));
		// free gpu lock
		free_gpu_lock<<<1, 1>>>();
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
	{
		Request Img_req;
		Img_req.imgID = img_id;
		Img_req.imgIn = img_in;
		Img_req.imgOut = img_out;  // Already allocated as shared host memory by the caller
		
		// must implement non blocking push for this to be non blocking
		return cpu_to_gpu_queue->push(Img_req);
	}

	bool dequeue(int *img_id) override
	{
		// TODO query (don't block) the producer-consumer queue for any responses.
		// must implement non blocking pop to support this
		Request Img_req = gpu_to_cpu_queue->pop();
		if (Img_req.imgID == INVALID_IMAGE)
		{
			return false;
		}

		// TODO return the img_id of the request that was completed.
		*img_id = Img_req.imgID;

		return true;
	}
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
	return std::make_unique<queue_server>(threads);
}
