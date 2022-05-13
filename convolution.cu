#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  /*if(j == 0 || j > 3*rows || i%(3*cols) == 0 || i > 3*cols){
    g[ j * cols + i ] = 0;

  }else{*/
     auto courant = rgb[(j * cols + i)];
     auto gauche = rgb[(j * cols +(i-1))];
     auto droite = rgb[(j * cols +(i+1))];
     auto bas = rgb[((j+1) * cols + i)];
     auto haut = rgb[((j-1) * cols + i)];
     auto hautGauche = rgb[((j-1) * cols + (i-1))];
     auto hautDroite = rgb[((j-1) * cols + (i+1))];
     auto basGauche = rgb[((j+1) * cols + (i-1))];
     auto basDroite = rgb[((j+1) * cols + (i+1))];
     int matrix[3][3] = {0, 0 , 0, 0, 1, 0, 0, 0, 0};
     int res = hautGauche * matrix[0][0] + haut * matrix[0][1] + hautDroite * matrix[0][2] + gauche * matrix[1][0] + courant * matrix[1][1] + droite * matrix[1][2] + basGauche * matrix[2][0] + bas * matrix[2][1] + basDroite *matrix[2][2];
     g[(j * cols + i)] = res;

}

int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  std::vector< unsigned char > g(3* rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC3, g.data() );
  unsigned char * rgb_d;
  unsigned char * g_d;
  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &g_d, 3* rows * cols );
  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  dim3 t( 32, 32 );
  dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );
  grayscale<<< b, t >>>( rgb_d, g_d, cols, rows );
  cudaMemcpy( g.data(), g_d, 3 * rows * cols, cudaMemcpyDeviceToHost );
  cv::imwrite( "out.jpg", m_out );
  cudaFree( rgb_d);
  cudaFree( g_d);
  return 0;
}
