#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>

using namespace std;

#define SIZE 100

void initialData(float ip[SIZE][SIZE])
{
    int i;
    int j;

    for(i = 0; i < SIZE; i++)
    {
        for(j = 0; j < SIZE; j++)
          ip[i][j] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void multMatrix(float A[SIZE][SIZE], float B[SIZE][SIZE], float C[SIZE][SIZE],
                  const int nx, const int ny)
{
  for (int iy = 0; iy < ny; iy++)
  {
    for (int ix = 0; ix < nx; ix++)
    {
      for (int im=0; im < nx; im++)
      {
          C[iy][ix] += A[iy][im] * B[im][ix];
      }
    }
  }

    return;
}

void printMatrix(int nx, int ny, float matrix[SIZE][SIZE])
{
  int row, columns;
  for (int row=0; row<nx; row++)
  {
      for(int columns=0; columns<ny; columns++)
          {
           printf("%f     ", matrix[row][columns]);
          }
      printf("\n");
   }
}

int main(int argc, char **argv)
{
  float h_A[SIZE][SIZE];
  float h_B[SIZE][SIZE];
  float hostRef[SIZE][SIZE];

  int nx = SIZE;
  int ny = SIZE;

  initialData(h_A);
  initialData(h_B);

  // multMatrix(h_A, h_B, hostRef, nx, ny);
  // printMatrix(nx, ny, h_A);
  // printf("\n");
  // // printMatrix(nx, ny, h_B);
  // printf("\n");
  // printMatrix(nx, ny, hostRef);
}
