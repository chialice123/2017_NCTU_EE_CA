// This program executes a typical convolutional layer in regular CNNs.Neuron sparsity(zero ratio) is 50% and Weight sparsity is 70%.
#include <iostream>
#include "CNNConvLayer.h"
using namespace std;

int *inGNeu, *inFilt, *outGNeu, *outGGPU; //address for GPU
//declare variables for GPU
void initGPU()
{   
	//allocate memory space(VRAM) on GPU
	cudaMalloc(&inGNeu, sizeof(int)* FMSIZE*FMSIZE*FMDEPTH );
	cudaMalloc(&inFilt, sizeof(int) * FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM );
	cudaMalloc(&outGNeu, sizeof(int)* FILTNUM * FMSIZE * FMSIZE );
	cudaMalloc(&outGGPU, sizeof(int)* FILTNUM * FMSIZE/3 * FMSIZE/3 );
	
	//copy data from DRAM to VRAM
	cudaMemcpy(inGNeu, inNeu, sizeof(int)* FMSIZE*FMSIZE*FMDEPTH, cudaMemcpyHostToDevice);
	cudaMemcpy(inFilt, filt, sizeof(int)* FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM, cudaMemcpyHostToDevice);
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
	int outArea  = FMSIZE/3 * FMSIZE/3;
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

	// Max Pooling with Window Size 3x3 and stride 3
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++){
		for(fmy = 0; fmy < FMSIZE/3 ; fmy += 1){
			for(fmx = 0; fmx < FMSIZE/3 ; fmx += 1){
				outNeuIdx = sli*fmArea + fmy*3*FMSIZE + fmx*3;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 3; y++){
					for(x = 0; x < 3; x++){
						ofmy = fmy*3 + y;
						ofmx = fmx*3 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];	
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE/3 + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}

/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU(int* inGNeu, int* inFilt, int* outGGPU, int* outGNeu) //nonshared
{
	//threads and blocks
	int bx = blockIdx.x; //FILTNUM 128	
	int tx = threadIdx.x; //FMSIZE 27 x(col)
	int ty = threadIdx.y; //FMSIZE 27 y(row)
	
	int sli;
	int ifmy, ifmx, ofmy, ofmx, x, y;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol  = FMDEPTH  * FILTSIZE * FILTSIZE; // 96 * 5 * 5
	int fmArea   = FMSIZE   * FMSIZE; // 27 * 27
	int filtArea = FILTSIZE * FILTSIZE; // 5 * 5
	int outArea  = FMSIZE/3 * FMSIZE/3;
	int sum = 0;
	
	//convolution
	for (sli = 0; sli < FMDEPTH; sli++){
		for(y = 0; y < FILTSIZE; y++){ // FILTSIZE 5 y(row)
			for(x = 0; x < FILTSIZE; x++){ // FILTSIZE 5 x(col)
				ifmy = ty - FILTSIZE / 2 + y; //frame_row - 5/2 + filter_row
				ifmx = tx - FILTSIZE / 2 + x; //frame_col - 5/2 + filter_col
				filtIdx = bx*filtVol + sli*filtArea + y*FILTSIZE + x;
				//in_idx = 第幾層*27*27 + frame(row*dim+col)
				inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
				//inside frame
				if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
					sum += inFilt[filtIdx] * inGNeu[inNeuIdx];
			}
		}
	}
	
	__syncthreads();
	
	outNeuIdx = bx*fmArea + ty*FMSIZE + tx;
	//reLu
	if(sum <= 0)
		outGNeu[outNeuIdx] = 0;
	else
		outGNeu[outNeuIdx] = sum;	
	
	__syncthreads();
	// Max Pooling with Window Size 3x3 and stride 3
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++){
		if (ty < FMSIZE/3){
			if (tx < FMSIZE/3){
				outNeuIdx = sli*fmArea + ty*3*FMSIZE + tx*3;
				max = outGNeu[outNeuIdx];
				for(y = 0; y < 3; y++){
					for(x = 0; x < 3; x++){
						ofmy = ty*3 + y;
						ofmx = tx*3 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outGNeu[outNeuIdx];	
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + ty*FMSIZE/3 + tx;
				outGGPU[outIdx] = max;
			}
		}
	}
}
/***	Implement your CUDA Kernel here	***/

int main()
{
	cudaSetDevice(2);
	//variables setting and loading input data
	timespec time_begin, time_end; 
	int convLayerCPUExecTime, convLayerGPUExecTime;
	init();
	
	//Convolution by CPU                                                
	clock_gettime(CLOCK_REALTIME, &time_begin);
	convLayerCPU();
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = "  <<  ((float)convLayerCPUExecTime)/1000 << "ms" << endl;

	dim3 numBlocks(FILTNUM); //128
	dim3 threadsPerBlock(FMSIZE,FMSIZE); //27*27
	
	//Convolution by GPU
	clock_gettime(CLOCK_REALTIME, &time_begin);
	/***	Lunch your CUDA Kernel here	***/
	initGPU();
	convLayerGPU<<<numBlocks,threadsPerBlock>>>(inGNeu, inFilt, outGGPU, outGNeu); // Lunch the kernel
	cudaDeviceSynchronize(); // Do synchronization before clock_gettime()
	cudaMemcpy(outGPU, outGGPU , sizeof(int) * FILTNUM * FMSIZE/3 * FMSIZE/3, cudaMemcpyDeviceToHost);
	/***	Lunch your CUDA Kernel here	***/
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = "  << ((float)convLayerGPUExecTime)/1000 << "ms" << endl;
	
	//cudaMemcpy(outNeu_, outGNeu, sizeof(int) * FILTNUM * FMSIZE * FMSIZE, cudaMemcpyDeviceToHost);
	
	cudaFree(&inGNeu);
	cudaFree(&inFilt);
	cudaFree(&outGGPU);
	cudaFree(&outGNeu);
	//check the anser from CPU and from GPU
	if(checker()){
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	//release memory space
	ending();
	
	return 0;
}
