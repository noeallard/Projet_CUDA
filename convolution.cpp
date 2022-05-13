#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>

int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto channels = m_in.channels();
  auto size = m_in.rows * m_in.cols * channels;
  auto cols = m_in.cols;
  auto rows = m_in.rows;

  std::vector< unsigned char > g( size );
  for (int j=0; j< rows-1 ; j++){


  for (int i=0; i< cols-1;i++){
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
        int matrix[3][3] = {1, 12 , 7, 1, -5,1, 1, -4, 1};
        int res = hautGauche * matrix[0][0] + haut * matrix[0][1] + hautDroite * matrix[0][2] + gauche * matrix[1][0] + courant * matrix[1][1] + droite * matrix[1][2] + basGauche * matrix[2][0] + bas * matrix[2][1] + basDroite * matrix[2][2];
        g[(j * cols + i) * channels + k] = res;
      }
    }
}
  }
  cv::Mat m_out( m_in.rows, m_in.cols, m_in.type(), g.data() );





  cv::imwrite( "out.jpg", m_out );

  return 0;
}

