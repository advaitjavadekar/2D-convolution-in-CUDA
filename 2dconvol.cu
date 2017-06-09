
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<fstream>
#include<iostream>
#include<sstream>
#include<cuda.h>
#include<cstdlib>
#include<string>

float a[3000][3000], h[10][10],c[3000][3000];
using namespace std;

//kernel function to run on a single thread
__global__ void conv2D(float *d_c, float *d_a, float *d_h,int arows,int acols,int hrows,int hcols)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int i, j;
	float sum;

	__shared__ float  shared_h[100];

	if (idx % 16 < hcols && idy % 16 < hrows) {
		int t = (idy % 16)*hcols + (idx % 16);
		shared_h[t] = d_h[t];
	}

	if (idy < (arows + hrows - 1) && idx < (acols + hcols - 1)) {
		sum = 0;
		for (i = 0; i < hrows; i++) {
			for (j = 0; j < hcols; j++) {
				if (!((idy - i) < 0 || (idx - j) < 0 || (idy - i) >= arows || (idx - j) >= acols)) {
					sum += d_a[((idy - i)*acols) + (idx - j)] * shared_h[i*hcols + j];
				}
			}
		}
		d_c[idy*(acols + hcols - 1) + idx] = sum;
	}
}

int main(int argc, char** argv)
{
	cudaError_t err = cudaSuccess;
	int acolst, arowst, hrowst, hcolst, a_elems, h_elems;
	string line;
	char* input_file;
	float test;
	int i, j;


	//read from file to get size of a and h
	input_file = argv[1];//"A:/i1024.txt";

	a_elems = 0;
	acolst = 0;
	arowst = 0;
	h_elems = 0;
	hcolst = 0;
	hrowst = 0;

	//read from file to get size of a and h
	ifstream file(input_file);
	if (file.is_open()) {
		i = 0;
		while (getline(file, line) && line != "") {
			j = 0;
			stringstream ss(line);
			while (ss >> test) {
				a[i][j] = test;
				j++;
				a_elems++;
			}
			i++;
			arowst++;
		}


		i = 0;
		while (getline(file, line) && line != "") {
			j = 0;
			stringstream ss(line);
			while (ss >> test) {
				h[i][j] = test;
				j++;
				h_elems++;
			}
			i++;
			hrowst++;
		}

	}
	file.close();

	acolst = a_elems / arowst;
	hcolst = h_elems / hrowst;

	const int acols = acolst;
	const int arows = arowst;
	const int hrows = hrowst;
	const int hcols = hcolst;

	//cout << acols << " " << arows << " " << hcols << " " << hrows << endl;
	
	//assign sizes to a,h,c
	float* h_a = new float[arows*acols];

	float* h_h = new float[hrows*hcols];

	float* h_c = new float[(arows+hrows-1)*(acols+hcols-1)];

	for (i = 0; i < arows; i++) {
		for (j = 0; j < acols; j++) {
			h_a[(i*acols) + j] = a[i][j];
			//cout << h_a[(i*acols)+j]<< " ";
		}
		//cout << endl;
	}
	//cout << endl;

	for (i = 0; i < hrows; i++) {
		for (j = 0; j < hcols; j++) {
			h_h[(i*hcols)+j] = h[i][j];
			//cout << h_h[(i*hcols)+j] << " ";

		}
		//cout << endl;
	}
	//cout << endl;

	size_t size_a = arows*acols * sizeof(float);
	size_t size_h = hrows*hcols * sizeof(float);
	size_t size_c = (arows + hrows - 1)*(acols + hcols - 1) * sizeof(float);

	cout << "So the file is being read" << endl;;
	//inputs and outputs on the host
	//done as global 

	//declare GPU memory pointers
	float *d_a, *d_h, *d_c;
	
	
	//allocate GPU memory
	err = cudaMalloc((void**)&d_a, size_a);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector a (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err=cudaMalloc((void**)&d_h, size_h);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector h (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err=cudaMalloc((void**)&d_c, size_c);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector c (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	
	
	
	//transfer the data to GPU
	err=cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy h_a to d_a (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_h, h_h, size_h, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy h_h to d_h (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	dim3 num_threadblocks(((arows + hrows - 2) / 16) + 1, ((acols + hcols - 2) / 16) + 1, 1);
	/*int x = (arows + hrows - 2) / 16) + 1;
	int y = (acols + hcols - 2) / 16) + 1;
	cout << "no of blocks launched" << x << "x" << y << endl;
	*/
	dim3 threads_per_block(16, 16, 1);


	cout << "Launching kernel";

	//launch kernel on GPU_
	conv2D <<<num_threadblocks+num_threadblocks%threads_per_block, threads_per_block >>> (d_c,d_a,d_h,arows,acols,hrows,hcols);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch conv2D kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err = cudaDeviceSynchronize();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}




	//transfer result to CPU
	cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_c to h_c (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}



	cout << "And it's done... Data is out of GPU and in CPU memory"<<endl;


/*
	//Calc conv2D on CPU
	kCenterX = hcols / 2;
	kCenterY = hrows / 2;

	for (i = 0; i < arows; ++i){
		for (j = 0; j < acols; ++j){
			for (m = 0; m < hrows; ++m){     // kernel rows
				mm = hrows - 1 - m;          // row index of flipped kernel
				for (n = 0; n < hcols; ++n){ // kernel columns
					nn = hcols - 1 - n;      // column index of flipped kernel		 
					ii = i + (m - kCenterY);
					jj = j + (n - kCenterX);// index of input signal, used for checking boundary

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < arows && jj >= 0 && jj < acols)
						h_c[i][j] += h_a[ii][jj] * h_h[mm][nn];

				}
			}
		}
	}*/

	//print output
	for (i = 0; i < arows+hrows-1; i++) {
		for (j = 0; j < acols+hcols-1; j++) {
			//h_h[i][j] = h[i][j];
			cout << h_c[(i*(acols+hcols-1))+j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	
	//free GPU memory location
	err = cudaFree(d_a);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err=cudaFree(d_h);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err=cudaFree(d_c);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	////Driver to reset all state
	//err = cudaDeviceReset();

	//if (err != cudaSuccess)
	//{
	//	fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
	//	exit(EXIT_FAILURE);
	//}
	
	return(0);
}

