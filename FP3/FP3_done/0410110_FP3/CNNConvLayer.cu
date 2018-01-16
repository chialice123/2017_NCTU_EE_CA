// This program executes a typical convolutional layer in regular CNNs.Neuron sparsity(zero ratio) is 50% and Weight sparsity is 70%.
#include <iostream>
#include "CNNConvLayer.h"
using namespace std;

int *inGNeu, *inFilt, *outGNeu, *outGGPU; //address for GPU
//declare variables for GPU
void initGPU()
{   
	cudaMalloc(&inGNeu, sizeof(int)* FMDEPTH * FMSIZE * FMSIZE );
	cudaMalloc(&inFilt, sizeof(int)*FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM );
	cudaMalloc(&outGNeu, sizeof(int)* FMSIZE * FMSIZE *FILTNUM  );
	cudaMalloc(&outGGPU, sizeof(int)* FMSIZE * FMSIZE *FILTNUM /4  );

	cudaMemcpy(inGNeu, inNeu, sizeof(int)* FMDEPTH * FMSIZE * FMSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(inFilt, filt, sizeof(int)*FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM, cudaMemcpyHostToDevice);
}

// This is the CPU version, please don't modify it
void convLayerCPU()
{
	// declarations for bunch of indexing parameters
	int fn, sli, fmy, fmx, y, x;
	int ifmy, ifmx, ofmy, ofmx;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol  = FMDEPTH  * FILTSIZE * FILTSIZE;
	int fmArea   = FMSIZE   * FMSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int outArea  = FMSIZE/2 * FMSIZE/2;
	int sum;
	// Convolution
	for(fn = 0; fn < FILTNUM; fn++){
		for(fmy = 0; fmy < FMSIZE; fmy += STRIDE){
			for(fmx = 0; fmx < FMSIZE; fmx += STRIDE){
				sum = 0;
				for(sli = 0; sli < FMDEPTH; sli++){
					for(y = 0; y < FILTSIZE; y++){
						for(x = 0; x < FILTSIZE; x++){
							ifmy = fmy - FILTSIZE / 2 + y;
							ifmx = fmx - FILTSIZE / 2 + x;
							filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
							if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
								sum += filt[filtIdx] * inNeu[inNeuIdx];
						}
					}
				}
				// Activation - ReLU
				outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
				if(sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = sum;
			}
		}
	}

	// Max Pooling with Window Size 2x2 and stride 2
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++){
		for(fmy = 0; fmy < FMSIZE/2 ; fmy += 1){
			for(fmx = 0; fmx < FMSIZE/2 ; fmx += 1){
				outNeuIdx = sli*fmArea + fmy*2*FMSIZE + fmx*2;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 2; y++){
					for(x = 0; x < 2; x++){
						ofmy = fmy*2 + y;
						ofmx = fmx*2 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];	
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE/2 + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}

/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU(int* inGNeu, int* inFilt, int* outGNeu) //nonshared
{
	//threads and blocks
	int fn = blockIdx.x; //FILTNUM 256
	int fmy = threadIdx.x; //FMSIZE 28 y(col)
	int fmx = threadIdx.y; //FMSIZE 28 x(row)
	
// declarations for bunch of indexing parameters
	int sli, y, x;
	int ifmy, ifmx;
	int filtIdx, inNeuIdx, outNeuIdx;
	int filtVol  = FMDEPTH  * FILTSIZE * FILTSIZE;
	int fmArea   = FMSIZE   * FMSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int sum = 0;
	// Convolution
	for(sli = 0; sli < FMDEPTH; sli++){
		for(y = 0; y < FILTSIZE; y++){
			for(x = 0; x < FILTSIZE; x++){
				ifmy = fmy - (FILTSIZE/2) + y;
				ifmx = fmx - (FILTSIZE/2) + x;
				filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
				inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
				if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
					sum += inFilt[filtIdx] * inGNeu[inNeuIdx];
			}
		}
	}
	// __syncthreads();
	// Activation - ReLU
	outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;

	if(sum <= 0)
		outGNeu[outNeuIdx] = 0;
	else
		outGNeu[outNeuIdx] = sum;
	// __syncthreads();
}

__global__ void PoolingGPU (int* outGGPU, int* outGNeu)
{
	//threads and blocks
	int sli = blockIdx.x; //FILTNUM 256
	int fmx = threadIdx.x; //FMSIZE/2 14 x(col)
	int fmy = threadIdx.y; //FMSIZE/2 14 y(row)
	
	int ofmy, ofmx, x, y;
	int outNeuIdx, outIdx;

	int fmArea   = FMSIZE   * FMSIZE; // 28 * 28
	int outArea  = FMSIZE/2 * FMSIZE/2;
	int max, tmpVal;
	
	outNeuIdx = sli*fmArea + fmy*2*FMSIZE + fmx*2;
	max = outGNeu[outNeuIdx];
	for(y = 0; y < 2; y++){
		for(x = 0; x < 2; x++){
			ofmy = fmy*2 + y;
			ofmx = fmx*2 + x;
			outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
			tmpVal = outGNeu[outNeuIdx];	
			if(tmpVal > max)
				max = tmpVal;
		}
	}
	__syncthreads();
	outIdx = sli*outArea + fmy*FMSIZE/2 + fmx;
	outGGPU[outIdx] = max;
	// __syncthreads();
}

__global__
void convGPU(int *inGNeu , int *inFilt , int *outGNeu)
{
	//threads and blocks
	int fn = blockIdx.x; //FILTNUM 256
	int fmy = blockIdx.y; //FMSIZE 28 y(col)
	int fmx = blockIdx.z; //FMSIZE 28 x(row)
	int sli = threadIdx.x;
	__shared__ int sum[FMDEPTH];
// declarations for bunch of indexing parameters
	int y, x;
	int ifmy, ifmx;
	int filtIdx, inNeuIdx, outNeuIdx;
	int filtVol  = FMDEPTH  * FILTSIZE * FILTSIZE;
	int fmArea   = FMSIZE   * FMSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int total_sum;
	sum[sli] = 0;
	// Convolution
	for(y = 0; y < FILTSIZE; y++){
		for(x = 0; x < FILTSIZE; x++){
			ifmy = fmy - FILTSIZE / 2 + y;
			ifmx = fmx - FILTSIZE / 2 + x;
			filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
			inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
			if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
				sum[sli] += inFilt[filtIdx] * inGNeu[inNeuIdx];
		}
	}
	__syncthreads();
	// Activation - ReLU
	if(threadIdx.x==0){
		total_sum =0;
		for(int p=0;p<FMDEPTH;p++)
			total_sum += sum[p];
			
		outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
		if(total_sum <= 0)
			outGNeu[outNeuIdx] = 0;
		else
			outGNeu[outNeuIdx] = total_sum;
	}
	// __syncthreads();
}

__global__
void poolGPU(int *inNeu , int *outNeu)
{	
	int Tx = threadIdx.x;
	int Ty = threadIdx.y;
    int block = blockIdx.x;
	int inNeu_ID = block*FMSIZE*FMSIZE + Ty*FMSIZE*2 + Tx*2 ;
	int Max = inNeu[inNeu_ID];

	if( Max < inNeu[inNeu_ID+1] )
		Max = inNeu[inNeu_ID+1];
	else
		Max = Max;
	 __syncthreads();
	if( Max < inNeu[inNeu_ID+FMSIZE] )
		Max = inNeu[inNeu_ID+FMSIZE];
	else
		Max = Max;
	 __syncthreads();
	if( Max < inNeu[inNeu_ID+FMSIZE+1] )
		Max = inNeu[inNeu_ID+FMSIZE+1];
	else
		Max = Max;

	 __syncthreads();
	outNeu[block*FMSIZE/2*FMSIZE/2 + Ty*FMSIZE/2 + Tx] = Max;
	
}

__global__
void convGPU_v1(int *inNeu , int *filt , int *outNeu)
{
    int fn, sli, fmy, fmx, y, x ,p ;
	int ifmy, ifmx;
	int filtIdx, inNeuIdx, outNeuIdx;
	int filtVol = FMDEPTH * FILTSIZE * FILTSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int fmArea = FMSIZE *FMSIZE;
    int total_sum;
	__shared__ int sum[FMDEPTH];
	int tile;
	
	sli=threadIdx.x;
	tile=128*blockIdx.x;
	fn=blockIdx.y;
	
	for(fmy = 0; fmy < FMSIZE; fmy += STRIDE){
		for(fmx = 0; fmx < FMSIZE; fmx += STRIDE){
			sum[sli] = 0;
			for(y = 0; y < FILTSIZE; y++){
				for(x = 0; x < FILTSIZE; x++){
					ifmy = fmy - FILTSIZE / 2 + y;
					ifmx = fmx - FILTSIZE / 2 + x;
					filtIdx = (fn+tile)*filtVol + sli*filtArea + y*FILTSIZE + x;
					inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
					
					if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE){
						sum[sli] += filt[filtIdx] * inNeu[inNeuIdx];
					}
						
				}
			}
			__syncthreads();
			
			if(threadIdx.x==0){
				total_sum =0;
				for(p=0;p<FMDEPTH;p++)
					total_sum=total_sum+sum[p];
				
				outNeuIdx = (fn+tile)*fmArea + fmy*FMSIZE + fmx;
				if(total_sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = total_sum;
			}
			__syncthreads();				
		}
	}
				
}

/***	Implement your CUDA Kernel here	***/

int main()
{
	//variables setting and loading input data
	timespec time_begin, time_end; 
	int convLayerCPUExecTime, convLayerGPUExecTime;
	init();
	initCoo();

	//Convolution by CPU                                              
	clock_gettime(CLOCK_REALTIME, &time_begin);
	convLayerCPU();
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = " <<  convLayerCPUExecTime / 1000 << "ms" << endl;

	//Initialize GPU
	initGPU();
	//Convolution by GPU   
	clock_gettime(CLOCK_REALTIME, &time_begin);
	/***	Lunch your CUDA Kernel here	***/
	/**** version 1 ****/
	// dim3 blocks(2,FILTNUM/2);
	// convGPU_v1<<<blocks,FMDEPTH>>>(inGNeu,inFilt,outGNeu); 
	// cudaDeviceSynchronize(); 
	// dim3 v1_threadsPerBlock_pool(FMSIZE/2,FMSIZE/2);
	// dim3 v1_numBlocks_pool(FILTNUM,1);
	// poolGPU<<<v1_numBlocks_pool,v1_threadsPerBlock_pool>>>(outGNeu,outGGPU);
	// cudaDeviceSynchronize(); 
	/**** w/ memory coalescing ****/
	// dim3 _numBlocks(FILTNUM,FMSIZE,FMSIZE); //256
	// dim3 _threadsPerBlock(FMDEPTH); //28*28
	// convGPU<<<_numBlocks,_threadsPerBlock>>>(inGNeu,inFilt,outGNeu); 
	// cudaDeviceSynchronize(); 
	// dim3 threadsPerBlock_pool(FMSIZE/2,FMSIZE/2);
	// dim3 numBlocks_pool(FILTNUM,1);
	// poolGPU<<<numBlocks_pool,threadsPerBlock_pool>>>(outGNeu,outGGPU);
	// cudaDeviceSynchronize();
	/**** final version ****/
	dim3 numBlocks(FILTNUM); //256
	dim3 threadsPerBlock(FMSIZE,FMSIZE); //28*28
	convLayerGPU<<<numBlocks,threadsPerBlock>>>(inGNeu, inFilt, outGNeu);
	cudaDeviceSynchronize();
	dim3 P_numBlocks(FILTNUM); //256
	dim3 P_threadsPerBlock(FMSIZE/2,FMSIZE/2); //14,14
	PoolingGPU<<<P_numBlocks,P_threadsPerBlock>>>(outGGPU, outGNeu);
	cudaDeviceSynchronize(); 
	/***	Lunch your CUDA Kernel here	***/
	clock_gettime(CLOCK_REALTIME, &time_end);
	cudaMemcpy(outGPU, outGGPU , sizeof(int) * FILTNUM * FMSIZE/2 * FMSIZE/2, cudaMemcpyDeviceToHost);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = "  << ((float)convLayerGPUExecTime)/1000 << "ms" << endl;
	
	cudaFree(&inGNeu);
	cudaFree(&inFilt);
	cudaFree(&outGGPU);
	cudaFree(&outGNeu);
	//check the anser from CPU and from GPU
	if(checker()){
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}else
		cout << "Sorry! Your result is wrong." << endl;

	//release memory space
	ending();
	
	return 0;
}