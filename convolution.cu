#include <opencv2/opencv.hpp>
#include <vector>

__global__ void flou( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows, int channels) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

     if( i < cols && j < rows ) {
       for (unsigned char k =0 ; k<channels ; k++){
         if (i==0 || j==0 || i==cols || j==rows){
           g[(j * cols + i)]=rgb[(j * cols + i)] = rgb[(j * cols + i)]=rgb[(j * cols + i)];
         }
         else{
           //g[(j * cols + i) * channels + k] = rgb[(j * cols + i) * channels + k];
           auto courant = rgb[(j * cols + i) * channels + k];
           auto gauche = rgb[(j * cols +(i-1)) * channels + k];
           auto droite = rgb[(j * cols +(i+1)) * channels + k];
           auto bas = rgb[((j+1) * cols + i) * channels + k];
           auto haut = rgb[((j-1) * cols + i) * channels + k];
           auto hautGauche = rgb[((j-1) * cols + (i-1)) * channels + k];
           auto hautDroite = rgb[((j-1) * cols + (i+1)) * channels + k];
           auto basGauche = rgb[((j+1) * cols + (i-1)) * channels + k];
           auto basDroite = rgb[((j+1) * cols + (i+1)) * channels + k];
           int matrix[3][3] = {1, 1 , 1, 1, 1,1, 1, 1, 1};
           int res = hautGauche * matrix[0][0] + haut * matrix[0][1] + hautDroite * matrix[0][2] + gauche * matrix[1][0] + courant * matrix[1][1] + droite * matrix[1][2] + basGauche * matrix[2][0] + bas * matrix[2][1] + basDroite * matrix[2][2];
           g[(j * cols + i) * channels + k] = res/9;
         }
       }
       //g[(j * cols + i)]=rgb[(j * cols + i)]
  }


}

__global__ void detectionContour( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows, int channels) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

     if( i < cols && j < rows ) {
       for (unsigned char k =0 ; k<channels ; k++){
         if (i==0 || j==0 || i==cols || j==rows){
           g[(j * cols + i)]=rgb[(j * cols + i)] = rgb[(j * cols + i)]=rgb[(j * cols + i)];
         }
         else{
           //g[(j * cols + i) * channels + k] = rgb[(j * cols + i) * channels + k];
           auto courant = rgb[(j * cols + i) * channels + k];
           auto gauche = rgb[(j * cols +(i-1)) * channels + k];
           auto droite = rgb[(j * cols +(i+1)) * channels + k];
           auto bas = rgb[((j+1) * cols + i) * channels + k];
           auto haut = rgb[((j-1) * cols + i) * channels + k];
           auto hautGauche = rgb[((j-1) * cols + (i-1)) * channels + k];
           auto hautDroite = rgb[((j-1) * cols + (i+1)) * channels + k];
           auto basGauche = rgb[((j+1) * cols + (i-1)) * channels + k];
           auto basDroite = rgb[((j+1) * cols + (i+1)) * channels + k];
           int matrix[3][3] = {0, 1 , 0, 1, -4,1, 0, 1, 0};
           int res = hautGauche * matrix[0][0] + haut * matrix[0][1] + hautDroite * matrix[0][2] + gauche * matrix[1][0] + courant * matrix[1][1] + droite * matrix[1][2] + basGauche * matrix[2][0] + bas * matrix[2][1] + basDroite * matrix[2][2];
           g[(j * cols + i) * channels + k] = res;
         }
       }
       //g[(j * cols + i)]=rgb[(j * cols + i)]
  }


}


int main()
{
  cudaError_t cudaStatus;
  cudaError_t kernelStatus;
  cudaEvent_t start, stop;

  cudaStatus = cudaEventCreate( &start );
  if (cudaStatus != cudaSuccess)
  {
	   std::cout << "Error CudaEventCreate start"  << " ";
  }
  cudaStatus = cudaEventCreate( &stop );
  if (cudaStatus != cudaSuccess)
  {
	   std::cout << "Error CudaEventCreate stop"  << " ";
  }
  cudaStatus = cudaEventRecord( start );
  if (cudaStatus != cudaSuccess)
  {
	   std::cout << "Error cudaEventRecord start"  << " ";
  }
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  auto channels = m_in.channels();
  auto size = rows * cols * channels;
  printf ("Channels: %d\n", channels);
  //std::vector< unsigned char > g( rows * cols );
  auto g = new unsigned char[ size];
  cv::Mat m_out( rows, cols, m_in.type(), g );
  unsigned char * rgb_d;
  unsigned char * g_d;

  cudaStatus = cudaMalloc( &rgb_d, size );
  if (cudaStatus != cudaSuccess)
  {
	   std::cout << "Error CudaMalloc rgb_d"  << " ";
  }

  cudaStatus = cudaMalloc( &g_d, size );
  if (cudaStatus != cudaSuccess)
  {
	   std::cout << "Error CudaMalloc g_d"  << " ";
  }

  cudaStatus = cudaMemcpy( rgb_d, rgb, size, cudaMemcpyHostToDevice );
  if (cudaStatus  != cudaSuccess)
  {
	  std::cout << "Error cudaMemcpy rgb -> rgb_d - HostToDevice" << " ";
  }

  dim3 t( 32, 32 );
  dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );

  flou<<< b, t >>>( rgb_d, g_d, cols, rows ,channels);
  kernelStatus = cudaGetLastError();
   if ( kernelStatus != cudaSuccess )
   {
	   std::cout << "CUDA Error"<< cudaGetErrorString(kernelStatus) << " ";
  }

  detectionContour<<< b, t >>>( g_d, g_d, cols, rows ,channels);
  kernelStatus = cudaGetLastError();
   if ( kernelStatus != cudaSuccess )
   {
	   std::cout << "CUDA Error"<< cudaGetErrorString(kernelStatus) << " ";
  }

  cudaStatus = cudaMemcpy( g, g_d, size, cudaMemcpyDeviceToHost );
  if (cudaStatus  != cudaSuccess)
  {
	  std::cout << "Error cudaMemcpy g_d -> g - DeviceToHost" << " ";
  }
  cv::imwrite( "out.jpg", m_out );
  cudaFree( rgb_d);
  cudaFree( g_d);
  cudaEventSynchronize( stop );
  cudaEventRecord( stop );
  float duration;
  cudaEventElapsedTime( &duration, start, stop );
  std::cout << "time=" << duration << std::endl;
  return 0;
}

