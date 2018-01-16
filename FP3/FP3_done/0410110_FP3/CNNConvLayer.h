#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

//important note：In this part, maxpooling is 2x2 

#define FMSIZE 28
#define FMDEPTH 192
#define FILTSIZE 3
#define FILTNUM 256
#define STRIDE 1

int*inNeu;
int*filt;
int*outNeu;
int*outCPU;
int*outGPU;

int *filtCooNNZ;
int *filtCooData;
int *filtCooRow;
int *filtCooCol;
int *tmp_filtCooData;
int *tmp_filtCooRow;
int *tmp_filtCooCol;

int *inNeuCooNNZ;
int *inNeuCooData;
int *inNeuCooRow;
int *inNeuCooCol;
int *tmp_inNeuCooData;
int *tmp_inNeuCooRow;
int *tmp_inNeuCooCol;


void init()
{
	ifstream ifs;
	string str;
	
	int inNeuIdx, filtIdx;
	int tmp;
	int outNeuVol = FILTNUM * FMSIZE * FMSIZE;
	int outVol = FILTNUM * FMSIZE/2 * FMSIZE/2;
	
	inNeu = new int[FMSIZE*FMSIZE*FMDEPTH]();
	ifs.open("data/neuron.txt", ifstream::in);
	if(!ifs.is_open()){
		cout << "Can not open the neurons input file\n";
	}
	
	for(int i = 0 ; i < FMDEPTH ; i++){
		ifs >> str; 
		for(int j = 0 ; j < FMSIZE ; j++){ 
			for(int k = 0 ; k < FMSIZE ; k++){ 
				ifs >> tmp;
				inNeuIdx = i*FMSIZE*FMSIZE + j*FMSIZE + k;
				inNeu[inNeuIdx] = tmp;
			}
		}
	}
	ifs.close();
				
		
	filt = new int[FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM]();
	ifs.open("data/filter.txt", ifstream::in);
	if(!ifs.is_open()){
		cout << "Can not open the filters input file\n";
	}
	
	for(int i=0 ; i<FILTNUM ; i++){ 
		for(int j = 0 ; j < FMDEPTH ; j++){	
			ifs >> str >> str >> str; 
			for(int k=0 ; k<FILTSIZE ; k++){
				for(int l=0 ; l<FILTSIZE ; l++){
					ifs >> tmp;
					filtIdx = i*FMDEPTH*FILTSIZE*FILTSIZE + j*FILTSIZE*FILTSIZE	+ k*FILTSIZE + l;
					filt[filtIdx] = tmp;
				}
			}
		}	
	}
	ifs.close();

	outNeu = new int[outNeuVol]();
	outCPU = new int[outVol]();
	outGPU = new int[outVol]();

}

void initCoo()
{
	int i, j, k, idx;
	int tmp, nnz;
	string str;
	int current_nnz;
	fstream ifs;

	filtCooNNZ = new int [FILTNUM*FMDEPTH];

	ifs.open("data/filt_COO_irregular.txt", ifstream::in);
	if(!ifs.is_open()){
		cout << "Can not open the filters input file\n";
		exit(-1);
	}
	current_nnz=0;
	
	
	for(i = 0; i < FILTNUM; i++){
		ifs >> str; 
		for(j = 0; j < FMDEPTH; j++){
			ifs >> str; 
			ifs >> str >> nnz; 
			idx = i*FMDEPTH + j;
			filtCooNNZ[idx] = nnz;
			
			if(i == 0 && j==0){
				filtCooData = new int [nnz];
				filtCooRow  = new int [nnz];
				filtCooCol  = new int [nnz];
			}
			else{
				tmp_filtCooData = new int [current_nnz];
				tmp_filtCooRow  = new int [current_nnz];
				tmp_filtCooCol  = new int [current_nnz];
			
				memcpy(tmp_filtCooData,filtCooData,sizeof(int)*current_nnz);
				memcpy(tmp_filtCooRow,filtCooRow,sizeof(int)*current_nnz);
				memcpy(tmp_filtCooCol,filtCooCol,sizeof(int)*current_nnz);
				
				delete [] filtCooData;
				delete [] filtCooRow;
				delete [] filtCooCol;
				
				filtCooData = new int [current_nnz+nnz];
				filtCooRow = new int [current_nnz+nnz];
				filtCooCol = new int [current_nnz+nnz];
				
				memcpy(filtCooData,tmp_filtCooData,sizeof(int)*current_nnz);
				memcpy(filtCooRow,tmp_filtCooRow,sizeof(int)*current_nnz);
				memcpy(filtCooCol,tmp_filtCooCol,sizeof(int)*current_nnz);
				
				
				delete [] tmp_filtCooData;
				delete [] tmp_filtCooRow;
				delete [] tmp_filtCooCol;
			}
			
			ifs >> str ;
			for(k = 0; k < nnz; k++){
				ifs >> tmp;
				idx = current_nnz + k;
				filtCooData[idx] = tmp;
				
			}
			ifs >> str ;
			for(k = 0; k < nnz; k++){
				ifs >> tmp;
				idx =current_nnz +k;
				filtCooRow[idx] = tmp;
			}
			ifs >> str ;
			for(k = 0; k < nnz; k++){
				ifs >>  tmp;
				idx = current_nnz+ k;
				filtCooCol[idx] = tmp;
			}
	
			// cout << "filtCooNNZ[" << i*FMDEPTH + j << "] =" << filtCooNNZ[i*FMDEPTH + j] << endl;
			// for(k = 0; k < nnz; k++){
				// cout << "filtCooData[" << current_nnz+k << "] =" << filtCooData[current_nnz+k] << endl;
				// cout << "filtCooRow[" << current_nnz+k << "] =" << filtCooRow[current_nnz+k] << endl;
				// cout << "filtCooCol[" << current_nnz+k << "] ="<< filtCooCol[current_nnz+k] << endl;
			// }
			
			current_nnz=current_nnz+nnz;
		
		}
	}
	ifs.close();
	
	
	

	current_nnz=0;
	inNeuCooNNZ = new int [FMDEPTH];

	ifs.open("data/neuron_COO_irregular.txt", ifstream::in);
	if(!ifs.is_open()){
		cout << "Can not open the neurons input file\n";
		exit(-1);
	}
	for(i = 0; i < FMDEPTH ; i++){
		ifs >> str; 
		ifs >> str >> nnz; 
		inNeuCooNNZ[i] = nnz;
		
		if(i == 0){
				inNeuCooData = new int [nnz];
				inNeuCooRow  = new int [nnz];
				inNeuCooCol  = new int [nnz];
		}
		else{
				tmp_inNeuCooData = new int [current_nnz];
				tmp_inNeuCooRow  = new int [current_nnz];
				tmp_inNeuCooCol  = new int [current_nnz];
			
				memcpy(tmp_inNeuCooData , inNeuCooData , sizeof(int)*current_nnz);
				memcpy(tmp_inNeuCooRow  , inNeuCooRow  , sizeof(int)*current_nnz);
				memcpy(tmp_inNeuCooCol  , inNeuCooCol  , sizeof(int)*current_nnz);
				
				delete [] inNeuCooData ;
				delete [] inNeuCooRow  ;
				delete [] inNeuCooCol  ;
				
				inNeuCooData = new int [current_nnz+nnz];
				inNeuCooRow  = new int [current_nnz+nnz];
				inNeuCooCol  = new int [current_nnz+nnz];
				
				memcpy(inNeuCooData , tmp_inNeuCooData ,sizeof(int)*current_nnz);
				memcpy(inNeuCooRow  , tmp_inNeuCooRow  ,sizeof(int)*current_nnz);
				memcpy(inNeuCooCol  , tmp_inNeuCooCol  ,sizeof(int)*current_nnz);
				
				delete [] tmp_inNeuCooData;
				delete [] tmp_inNeuCooRow;
				delete [] tmp_inNeuCooCol;
		}

		ifs >> str;
		for(j = 0; j < nnz; j++){
			ifs >> tmp;
			idx = current_nnz + j;
			inNeuCooData[idx] = tmp;
		}
		ifs >> str;
		for(j = 0; j < nnz; j++){
			ifs >> tmp;
			idx = current_nnz + j;
			inNeuCooRow[idx] = tmp;
		}
		ifs >> str;
		for(j = 0; j < nnz; j++){
			ifs >> tmp;
			idx = current_nnz + j;
			inNeuCooCol[idx] = tmp;
		}
		
		// cout << "inNeuCooNNZ[" << i << "] =" << inNeuCooNNZ[i] << endl;
		// for(k = 0; k < nnz; k++){
			// cout << "inNeuCooData[" << current_nnz+k << "] =" << inNeuCooData[current_nnz+k] << endl;
			// cout << "inNeuCooRow["  << current_nnz+k << "] =" << inNeuCooRow[current_nnz+k]  << endl;
			// cout << "inNeuCooCol["  << current_nnz+k << "] =" << inNeuCooCol[current_nnz+k]  << endl;
		// }
		
		current_nnz=current_nnz+nnz;
		
	}
	
	ifs.close();
	
	
	
	
}

void ending()
{
	delete [] filt;
	delete [] inNeu;
	delete [] outNeu;
	delete [] outCPU;
	delete [] outGPU;
	
	delete [] filtCooNNZ;
	delete [] filtCooData;
	delete [] filtCooRow;
	delete [] filtCooCol;


	delete [] inNeuCooNNZ;
	delete [] inNeuCooData;
	delete [] inNeuCooRow;
	delete [] inNeuCooCol;

	
}

bool checker(){
	int outVol = FILTNUM * FMSIZE/2 * FMSIZE/2;

	for(int i = 0; i < outVol; i++){ 
		if(  outCPU[i] != outGPU[i]   ){
			cout << "The element: " << i << " is wrong!\n";
			cout << "outCPU[" << i << "] = " << outCPU[i] << endl;
			cout << "outGPU[" << i << "] = " << outGPU[i] << endl;
			return false;
		}
	}
	return true;
}

int timespec_diff_us(timespec& t1, timespec& t2)
{                                                                                
  return (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_nsec - t1.tv_nsec) / 1e3;        
}