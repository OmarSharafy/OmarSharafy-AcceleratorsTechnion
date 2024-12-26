#include "ex1.h"
#define TILE_SIZE (TILE_WIDTH * TILE_WIDTH)
#define TILE_PER_IMAGE (TILE_COUNT * TILE_COUNT)
#define GREY_LEVELS 256
#define THREADS_PER_BLOCK 1024


#include "ex1.h"


__device__
void get_histogram(int* hist, uchar* all_in, int tile_row, int tile_col)
{
	const int tid = threadIdx.x;

	const int thread_work = TILE_WIDTH * TILE_WIDTH / blockDim.x; //thread work per Tile
	const int threads_per_row = TILE_WIDTH / thread_work;

	const int x_index = (TILE_WIDTH * tile_row) + (tid / threads_per_row);
	const int y_index = (TILE_WIDTH * tile_col) + ((tid % threads_per_row));

	if (x_index > (TILE_WIDTH * (tile_row + 1) ))
	{
		return; //no need to check y, but might need in special cases to mark that we were in this ?
		//not really -> same Y for diff tid will cause diff x
	}

	
	int color_value = 0;
	int index = 0;

	const int indexes_in_raw = TILE_COUNT * TILE_WIDTH;

	for(int j = 0 ; j < thread_work ; j ++)
	{
		index = x_index * indexes_in_raw + y_index + threads_per_row*j;
		color_value = all_in[index];
		atomicAdd(&hist[color_value], 1);
	}  
}

//Like the Toturial 
__device__
void prefix_sum(int arr[], int arr_size) 
{
	const int tid = threadIdx.x; 
	int increment;

	for (int stride = 1 ; stride < blockDim.x ; stride *= 2)
	{
		if (tid >= stride)
		{
			increment = arr[tid - stride];
		}

		__syncthreads();

		if (tid >= stride)
		{
			arr[tid] += increment;
		}
		__syncthreads();
	}
}


//Each thread should be assainged to one level of color
__device__
void get_maps(int* cdf, uchar* maps, int tile_row, int tile_col)
{
	const int tid = threadIdx.x;
	
	if (tid > GREY_LEVELS - 1)
	{
		// this the case more threads than we need 
		return;
	}

	const int maps_start_index = ((tile_row * TILE_COUNT) + tile_col) * GREY_LEVELS;
	maps[maps_start_index + tid] = (float(cdf[tid]) * (GREY_LEVELS - 1)) / (TILE_SIZE);
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
void interpolate_device(uchar* maps , uchar *in_img, uchar* out_img);

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) 
{
	const int hist_size = GREY_LEVELS * sizeof(int);
	__shared__ int hist[hist_size];

	int tid = threadIdx.x;
	
	//only one block work on each image
	const int image_offset = IMG_HEIGHT * IMG_WIDTH * blockIdx.x;
	const int maps_offset = GREY_LEVELS * TILE_COUNT * TILE_COUNT * blockIdx.x;


	for (int tile_row = 0 ; tile_row < TILE_COUNT ; tile_row++)
	{
		for (int tile_col = 0 ; tile_col < TILE_COUNT ; tile_col++)
		{

			for(int i=0; i < hist_size; i+=blockDim.x)
			{
				hist[tid+i] = 0;
			}
			__syncthreads();

			get_histogram(hist, all_in + image_offset, tile_row, tile_col);
			__syncthreads();          
	
			prefix_sum(hist, GREY_LEVELS); 
			__syncthreads();            

			get_maps(hist, maps + maps_offset, tile_row, tile_col);
			__syncthreads();
			
		}
	}
	
	interpolate_device(maps + maps_offset, all_in + image_offset, all_out + image_offset);
	__syncthreads();    

	return; 
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context 
{
	uchar* in_img;
	uchar* maps;
	uchar* out_img;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
	auto context = new task_serial_context;

	cudaMalloc(&context->in_img, IMG_HEIGHT * IMG_WIDTH * sizeof(uchar));
	cudaMalloc(&context->maps, TILE_COUNT * TILE_COUNT * GREY_LEVELS * sizeof(uchar));
	cudaMalloc(&context->out_img, IMG_HEIGHT * IMG_WIDTH * sizeof(uchar));

	return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
	for (int i = 0 ; i < N_IMAGES ; i++)
	{
		uchar* cur_images_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
		uchar* cur_images_out = &images_out[i * IMG_WIDTH * IMG_HEIGHT];
		CUDA_CHECK(cudaMemcpy(context->in_img, cur_images_in, IMG_HEIGHT * IMG_WIDTH * sizeof(uchar), cudaMemcpyHostToDevice));

		process_image_kernel<<<1, THREADS_PER_BLOCK>>>(context->in_img, context->out_img, context->maps);

		CUDA_CHECK(cudaMemcpy(cur_images_out, context->out_img, IMG_HEIGHT * IMG_WIDTH * sizeof(uchar), cudaMemcpyDeviceToHost));
	}
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
	CUDA_CHECK(cudaFree(context->in_img));
	CUDA_CHECK(cudaFree(context->out_img));
	CUDA_CHECK(cudaFree(context->maps));

	free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context 
{
	uchar* in_imgs;
	uchar* out_imgs;
	uchar* maps;
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
	auto context = new gpu_bulk_context;

	CUDA_CHECK(cudaMalloc(&context->in_imgs, N_IMAGES * IMG_HEIGHT * IMG_WIDTH * sizeof(uchar)));
	CUDA_CHECK(cudaMalloc(&context->out_imgs, N_IMAGES * IMG_HEIGHT * IMG_WIDTH * sizeof(uchar)));
	CUDA_CHECK(cudaMalloc(&context->maps, N_IMAGES * TILE_COUNT * TILE_COUNT * GREY_LEVELS * sizeof(uchar)));

	return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
	//TODO: copy all input images from images_in to the GPU memory you allocated
	//TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
	//TODO: copy output images from GPU memory to images_out
	CUDA_CHECK(cudaMemcpy(context->in_imgs, images_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH * sizeof(uchar), cudaMemcpyHostToDevice));
	// invoke kernel here
	process_image_kernel<<<N_IMAGES, THREADS_PER_BLOCK>>>(context->in_imgs, context->out_imgs, context->maps);
	CUDA_CHECK(cudaMemcpy(images_out, context->out_imgs, N_IMAGES * IMG_HEIGHT * IMG_WIDTH * sizeof(uchar), cudaMemcpyDeviceToHost));

}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
	CUDA_CHECK(cudaFree(context->in_imgs));
	CUDA_CHECK(cudaFree(context->out_imgs));
	CUDA_CHECK(cudaFree(context->maps));

	free(context);
}
