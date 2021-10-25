#include <string>
#include <iostream>
#include <stdlib.h>
#include <chrono>


#define p(I, J) *((p + (I)*n) + (J))

//######################################################################################
//####                                                                              ####
//####                               User Input (macros)                            ####
//####                                                                              ####
//######################################################################################

#define N 16                    // Chromosome size (number of genes per individual).
#define M 8                     // Population size (number of individuals).
//#define blockSize 8                     // Population size (number of individuals)
#define seed 42                 // Psuedorandom number generator seed (std:srand(seed)).
#define maxgenerations 2000     // Maximum number of generations (while loop limit).
#define t 2                     // Tournament size (parents competing for selection).
#define verbose true            // Verbose output (cout) for verification.
#define printfitness false      // Best fitness per generation output (cout).

//######################################################################################
//####                                                                              ####
//####                         DO NOT MODIFY BELOW THIS LINE                        ####
//####                                                                              ####
//######################################################################################

// global variables to store the matrix

int* P = nullptr;
int* F = nullptr;
int* pDevice;
int* fDevice;

void PrintPopulationVerbose(int *p, int n, int m)
/*
Fill static 2d int array at pointer *p with m indivduals with n
chromosmomes, gene values 0 or 1 (verbose cout).
*/
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << p(i, j);
        }
        std::cout << " -> Individual " << i << std::endl;
    }
    return;
}



void RandomPopulationVerbose(int *p, int n, int m)
/*
Fill static 2d int array at pointer *p with m indivduals with n
chromosmomes, gene values 0 or 1 (verbose cout).
*/
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            p(i, j) = ((int)std::rand() % 2);
            std::cout << p(i, j);
        }
        std::cout << " -> Individual " << i << std::endl;
    }
    return;
}


int MaxFitnessVerbose(int *p, int n, int m)
/*
Return integer of best (highest) fitness for the pointer *p with m indivduals with n
chromosmomes. Fitness is defined as the sum of the individuals chromosomes (verbose cout).
*/
{
    int best = 0;
    int besti = 0;
    int sum = 0;
    std::cout << "Running MaxFitness (Verbose)" << std::endl;
    for (int i = 0; i < m; i++)
    {
        sum = 0;
        for (int j = 0; j < n; j++)
        {
            std::cout << p(i, j);
            sum += p(i, j);
        }
        std::cout << " -> Individual " << i << ", Fitness: " << sum << std::endl;

        if (sum > best)
        {
            best = sum;
            besti = i;
        }
    }
    return best;
}



void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}


template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
extern __shared__ int sdata[];
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockSize*2) + tid;
unsigned int gridSize = blockSize*2*gridDim.x;
sdata[tid] = 0;
while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
__syncthreads();
if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
if (tid < 32) warpReduce<blockSize>(sdata, tid);
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}






int main()
{
    std::srand(seed);
    P = static_cast<int*>(malloc(N*M*sizeof(int)));
    F = static_cast<int*>(malloc(N*M*sizeof(int)));
    RandomPopulationVerbose(P, N,  M);
    MaxFitnessVerbose(P, N,  M);

    checkError(cudaMalloc(&pDevice, N*M*sizeof(int)));
    checkError(cudaMalloc(&fDevice, N*M*sizeof(int)));
    checkError(cudaMemcpy(pDevice, P, sizeof(double)*N*M, cudaMemcpyHostToDevice));

    int ThreadsPerBlock = 256;
    int Blocks = (N+ThreadsPerBlock-1)/ThreadsPerBlock;
    reduce6<4><<<Blocks,ThreadsPerBlock>>>(pDevice, fDevice, N);
    checkError(cudaMemcpy(F, fDevice, N*sizeof(double), cudaMemcpyDeviceToHost));
    PrintPopulationVerbose(F, N,  M);
    free(P);
    free(F);
    checkError(cudaFree(pDevice));
    checkError(cudaFree(fDevice));
    return 0;
}

