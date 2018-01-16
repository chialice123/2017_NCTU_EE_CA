// This program executes a typical convolutional layer in regular CNNs.Neuron sparsity(zero ratio) is 50% and Weight sparsity is 70%.
#include <iostream>
#include "CNNConvLayer.h"
#include <stdio.h>
#include <unistd.h>
using namespace std;


int outSize = sizeof(int)*128*9*9;
//int Outputsize = 128*27*27;

int *devoutNeu;
int *devPooling;
int *devFilt;
int *devinNeu;
int *inFilt;
 
/*COO Format*/
int *devfiltCooNNZ;
int *devfiltCooData;
int *devfiltCooRow;
int *devfiltCooCol;


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




void initGPU()
{
	int outNeuVol = FILTNUM * FMSIZE * FMSIZE;  		//128*27*27
	int outPolVol = FILTNUM * FMSIZE/3 * FMSIZE/3;  	//128*9*9 
	
	//output from kernel 
	cudaMalloc(&devoutNeu, sizeof(int)*outNeuVol);	//int
	cudaMalloc(&devPooling, sizeof(int)*outPolVol);	//int
	
	//input to kernel
	cudaMalloc(&devinNeu, sizeof(int)* FMSIZE*FMSIZE*FMDEPTH );
	cudaMalloc(&inFilt, sizeof(int) * FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM );
	
	cudaMemcpy(devinNeu, inNeu, sizeof(int)*FMSIZE*FMSIZE*FMDEPTH, cudaMemcpyHostToDevice);
	cudaMemcpy(inFilt, filt, sizeof(int)* FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM, cudaMemcpyHostToDevice);

}


void initCooMemoryCopy()
{
	int filtCOOVol = sizeof(int)*91306; 	
	int outNeuVol = FILTNUM * FMSIZE * FMSIZE;  		//128*27*27
	int outPolVol = FILTNUM * FMSIZE/3 * FMSIZE/3;  	//128*9*9 
	
	//output from kernel 
	cudaMalloc(&devoutNeu, sizeof(int)*outNeuVol);	
	cudaMalloc(&devPooling, sizeof(int)*outPolVol);	
		
	cudaMalloc(&devinNeu, sizeof(int)*FMGSIZE*FMGSIZE*FMDEPTH);	//input to kernel
	cudaMemcpy(devinNeu, inGNeu, sizeof(int)*FMGSIZE*FMGSIZE*FMDEPTH, cudaMemcpyHostToDevice);
	
	//input COO to kernel filter
	cudaMalloc(&devfiltCooNNZ, sizeof(int)*FILTNUM*FMDEPTH);
	cudaMalloc(&devfiltCooData, filtCOOVol);
	cudaMalloc(&devfiltCooRow, filtCOOVol);
	cudaMalloc(&devfiltCooCol, filtCOOVol);

	cudaMemcpy(devfiltCooNNZ, filtCooNNZ, sizeof(int)*FILTNUM*FMDEPTH, cudaMemcpyHostToDevice );
	cudaMemcpy(devfiltCooData, filtCooData, filtCOOVol, cudaMemcpyHostToDevice );
	cudaMemcpy(devfiltCooRow, filtCooRow, filtCOOVol, cudaMemcpyHostToDevice );
	cudaMemcpy(devfiltCooCol, filtCooCol, filtCOOVol, cudaMemcpyHostToDevice );

}



/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPUSparse(int *InNeu, int *FiltCooData, int *FiltCooRow, int *FiltCooCol, int *FiltCooNNZ, int *outNeural)
{
	// int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	// int threadY = threadIdx.y + blockIdx.y * blockDim.y;
	// int threadZ = threadIdx.z + blockIdx.z * blockDim.z;
	
	//int bx = blockIdx.x; //FILTNUM 128	
	int tx = threadIdx.x; //FMSIZE 27 x(col)
	int ty = threadIdx.y; 
	
	int ifmy, ifmx;
	int inNeuIdx, outNeuIdx, CooIdx;
	int sum = 0;

	int tmp = 0;
	for(int fn = 0 ; fn < 128 ; fn ++){
		sum=0;
		for(int sli = 0; sli < 96; sli++){
			for(int idx = 0 ; idx < FiltCooNNZ[fn*96+sli] ; idx++){
				CooIdx = tmp + idx;
				ifmx = tx  + FiltCooCol[CooIdx]; //col
				ifmy = ty  + FiltCooRow[CooIdx]; //row
				inNeuIdx = sli * FMGSIZE * FMGSIZE+ ifmy * FMGSIZE + ifmx;
				
				sum += FiltCooData[CooIdx] * InNeu[inNeuIdx];
				__syncthreads();				
			}
			tmp = FiltCooNNZ[fn*96 + sli] + tmp;
		}
	
	outNeuIdx = fn * FMSIZE * FMSIZE + ty*FMSIZE + tx;
	if(sum <= 0)
		outNeural[outNeuIdx] = 0;
	else
		outNeural[outNeuIdx] = sum;
	}
}


__global__
void convLayerGPU(int* inGNeu, int* inFilt, int* outGNeu) //nonshared
{
	//threads and blocks
	int bx = blockIdx.x; //FILTNUM 128	
	int tx = threadIdx.x; //FMSIZE 27 x(col)
	int ty = threadIdx.y; //FMSIZE 27 y(row)
	
	int sli;
	int ifmy, ifmx, x, y;
	int filtIdx, inNeuIdx, outNeuIdx;
	int filtVol  = FMDEPTH  * FILTSIZE * FILTSIZE; // 96 * 5 * 5
	int fmArea   = FMSIZE   * FMSIZE; // 27 * 27
	int filtArea = FILTSIZE * FILTSIZE; // 5 * 5
	int sum = 0;
	
	//convolution
	for (sli = 0; sli < FMDEPTH; sli++){
		for(y = 0; y < FILTSIZE; y++){ // FILTSIZE 5 y(row)
			for(x = 0; x < FILTSIZE; x++){ // FILTSIZE 5 x(col)
				ifmy = ty - FILTSIZE / 2 + y; //frame_row - 5/2 + filter_row
				ifmx = tx - FILTSIZE / 2 + x; //frame_col - 5/2 + filter_col
				filtIdx = bx*filtVol + sli*filtArea + y*FILTSIZE + x;
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
}


__global__
void MaxPoolingGPU(int *outGNeu, int *outGGPU) 
{
	// int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	// int threadY = threadIdx.y + blockIdx.y * blockDim.y;
	// int threadZ = threadIdx.z + blockIdx.z * blockDim.z;

	int bx = blockIdx.x; //FILTNUM 128
	int fmx = threadIdx.x; //FMSIZE/3 9 x(col)
	int fmy = threadIdx.y; //FMSIZE/3 9 y(row)
	
	int ofmy, ofmx, x, y;
	int outNeuIdx, outIdx;

	int fmArea   = FMSIZE   * FMSIZE; // 27 * 27
	int outArea  = FMSIZE/3 * FMSIZE/3;
	// Max Pooling with Window Size 3x3 and stride 3
	int max, tmpVal;
	outNeuIdx = bx*fmArea + fmy*3*FMSIZE + fmx*3;
	max = outGNeu[outNeuIdx];
	for(y = 0; y < 3; y++){
		for(x = 0; x < 3; x++){
			ofmy = fmy*3 + y;
			ofmx = fmx*3 + x;
			outNeuIdx = bx*fmArea + ofmy*FMSIZE + ofmx;
			tmpVal = outGNeu[outNeuIdx];	
			if(tmpVal > max)
				max = tmpVal;
		}
	}
	
	__syncthreads();
	outIdx = bx*outArea + fmy*FMSIZE/3 + fmx;
	outGGPU[outIdx] = max;
	__syncthreads();
}


/***	Implement your CUDA Kernel here	***/

int main()
{
	
	cudaSetDevice(2);
	cudaFree(0);
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
	cout << "CPU time for executing a typical convolutional layer = "  <<  ((float)convLayerCPUExecTime)/1000 << "ms" << endl;

  
	//Convolution by GPU   

	/***	Lunch your CUDA Kernel here	***/

	// original
	dim3 numBlocks(FILTNUM); //128
	dim3 threadsPerBlock(FMSIZE,FMSIZE); //27*27
	
	// Sparse CNN
	// dim3 _numBlocks(1);
	// dim3 _threadsPerBlock(FMSIZE,FMSIZE);
	
	// Max Pooling
	dim3 Pool_numBlocks(FILTNUM); //128
	dim3 Pool_threadPerBlock(FMSIZE/3,FMSIZE/3); //9,9

 	clock_gettime(CLOCK_REALTIME, &time_begin);

 	initGPU();
 	// initCooMemoryCopy();
 
	// convLayerGPUSparse<<<_numBlocks,_threadsPerBlock>>>(devinNeu, devfiltCooData, devfiltCooRow, devfiltCooCol, devfiltCooNNZ, devoutNeu);
	convLayerGPU<<<numBlocks,threadsPerBlock>>>(devinNeu, inFilt, devoutNeu);
	cudaDeviceSynchronize();
	
	MaxPoolingGPU<<<Pool_numBlocks , Pool_threadPerBlock>>>(devoutNeu, devPooling);
	cudaDeviceSynchronize();
	
	cudaMemcpy(outGPU, devPooling, outSize, cudaMemcpyDeviceToHost);

	/***	Lunch your CUDA Kernel here	***/
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = "  << ((float)convLayerGPUExecTime)/1000 << "ms" << endl;

	
	//check the anser from CPU and from GPU
	if(checker()){
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	//release memory space
	cudaFree(&devoutNeu);
	cudaFree(&devPooling);
	cudaFree(&devinNeu);

	cudaFree(&devfiltCooNNZ);
	cudaFree(&devfiltCooData);
	cudaFree(&devfiltCooRow);
	cudaFree(&devfiltCooCol);
	cudaFree(&inFilt);


	ending();
	
	return 0;
}
