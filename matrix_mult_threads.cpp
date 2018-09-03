#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include "omp.h"

using namespace std;

#define SIZE 4000

void initialData(int *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (int)(rand() & 0xFF);
    }

    return;
}

void printMatrix(int *A, const int nx, const int ny)
 {
     int *ia = A;

     for (int iy = 0; iy < ny; iy++)
     {
         for (int ix = 0; ix < nx; ix++)
         {
           printf("%d     ", ia[ix]);
         }
         printf("\n");
         ia += nx;
     }

     return;
 }

void multMatrix(int *A, int *B, int *C, const int nx,
                     const int ny)
{
    int *ia = A;
    int *ib = B;
    int *ic = C;

    int *ibm = B;

    int iy, ix, im;

    #pragma omp parallel for private(iy, ix, im, ibm) shared(ia, ib, ic)
      for (iy = 0; iy < ny; iy++)
      {
          for (ix = 0; ix < nx; ix++)
          {
              ibm = B;
              for (im = 0; im < nx; im++)
              {
                  ic[ix] += ia[im] * ibm[ix];
                  ibm += nx;
              }
          }

          ia += nx;
          ib += nx;
          ic += nx;
      }
}

int main(int argc, char **argv)
{
  int nx = SIZE;
  int ny = SIZE;

  int nxy = nx * ny;
  int nBytes = nxy * sizeof(int);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  int *h_A, *h_B, *hostRef;
  h_A = (int *)malloc(nBytes);
  h_B = (int *)malloc(nBytes);
  hostRef = (int *)malloc(nBytes);

  initialData(h_A, nxy);
  initialData(h_B, nxy);

  omp_set_num_threads(6);

  auto start_cpu =  chrono::high_resolution_clock::now();
  multMatrix(h_A, h_B, hostRef, nx, ny);
  auto end_cpu =  chrono::high_resolution_clock::now();
  chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

  printf("multMatrixOnHost elapsed %f ms\n", duration_ms.count());

  free(h_A);
  free(h_B);
  free(hostRef);
}
