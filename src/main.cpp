// g++ screena.cpp -o screena -lX11 -lXext -Ofast -mfpmath=both -march=native -m64 -funroll-loops -mavx2 `pkg-config opencv --cflags --libs` && ./screena

//#include <tesseract/baseapi.h>

//#include <opencv2/opencv.hpp>  // This includes most headers!

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <X11/extensions/XShm.h>
#include <sys/ipc.h>
#include <sys/shm.h>


#include <cstdint>
#include <vector>

#include <fstream>
#include <iostream>

#include "impl/bk_tree.hpp"
#include "impl/levenshtein_distance.hpp"


#include <time.h>
#define FPS(start) (CLOCKS_PER_SEC / (clock()-start))

#include<assert.h>

#include "impl/delay.hpp"

#include <tesseract/baseapi.h>
// #include <allheaders.h>
#include <sys/time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using std::string;
using std::vector;

#include <locale>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <cstring>
#include <cctype> // For isdigit()

std::string extract_ints(std::ctype_base::mask category, std::string str, std::ctype<char> const& facet) {
  using std::strlen;

  char const *begin = &str.front();
  char const *end   = &str.back();
  int i;
  for (i = 0; i < str.length(); i++) {
    if (! isdigit(str[i])) {
      if(i != 0) {
        //str = str.substr(0,i);
        //end = &str.back();
        break;
      }
    }
  }
  printf("End: %c \n", *end);

#if 1 // Insert whitespace between int and string
  str.insert(i, 1, ' ');

#else // Remove everything after int
  auto res = facet.scan_is(category, begin, end);
      assert(strlen(res) <= str.length());

  begin = &res[0];
  end   = &res[strlen(res)];

  std::cout << res << std::endl;
#endif

  return std::string(begin, end);
}

std::string extract_ints(std::string str) {
  return extract_ints(std::ctype_base::digit, str,
                      std::use_facet<std::ctype<char>>(std::locale("")));
}


int levDistance(const std::string source, const std::string target) {
  // Step 1
  const int n = source.length();
  const int m = target.length();
  if (n == 0) {
    return m;
  }
  if (m == 0) {
    return n;
  }
  // Good form to declare a TYPEDEF
  typedef std::vector< std::vector<int> > Tmatrix;
  Tmatrix matrix(n+1);
  // Size the vectors in the 2.nd dimension. Unfortunately C++ doesn't
  // allow for allocation on declaration of 2.nd dimension of vec of vec
  for (int i = 0; i <= n; i++) {
    matrix[i].resize(m+1);
  }
  // Step 2
  for (int i = 0; i <= n; i++) {
    matrix[i][0]=i;
  }
  for (int j = 0; j <= m; j++) {
    matrix[0][j]=j;
  }
  // Step 3
  for (int i = 1; i <= n; i++) {
    const char s_i = source[i-1];
    // Step 4
    for (int j = 1; j <= m; j++) {
      const char t_j = target[j-1];
      // Step 5
      int cost;
      if (s_i == t_j) {
        cost = 0;
      }
      else {
        cost = 1;
      }
      // Step 6
      const int above = matrix[i-1][j];
      const int left = matrix[i][j-1];
      const int diag = matrix[i-1][j-1];
      int cell = min( above + 1, min(left + 1, diag + cost));
      // Step 6A: Cover transposition, in addition to deletion,
      // insertion and substitution. This step is taken from:
      // Berghel, Hal ; Roach, David : "An Extension of Ukkonen's
      // Enhanced Dynamic Programming ASM Algorithm"
      // (http://www.acm.org/~hlb/publications/asm/asm.html)
      if (i>2 && j>2) {
        int trans=matrix[i-2][j-2]+1;
        if (source[i-2]!=t_j) trans++;
        if (s_i!=target[j-2]) trans++;
        if (cell>trans) cell=trans;
      }
      matrix[i][j]=cell;
    }
  }
  // Step 7
  return matrix[n][m];
}

#define INPUT_FILE              "sample.png"
#define OUTPUT_FOLDER_PATH      string("")

vector<Rect> findLetterContours(cv::Mat img) {
  //Mat large = imread(INPUT_FILE);
  Mat large = img.clone();
  Mat rgb;
  // downsample and use it for processing
  //pyrDown(large, rgb);
  rgb = large.clone();
  Mat small;
  cvtColor(rgb, small, CV_BGR2GRAY);
  // morphological gradient
  Mat grad;
  Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
  morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
  // binarize
  Mat bw;
  threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
  // connect horizontally oriented regions
  Mat connected;
  morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
  morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
  // find contours
  Mat mask = Mat::zeros(bw.size(), CV_8UC1);
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

  vector<Rect> vRects;//(contours.size());

  // filter contours
  for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
    {
      Rect rect = boundingRect(contours[idx]);
      Mat maskROI(mask, rect);
      maskROI = Scalar(0, 0, 0);
      // fill the contour
      drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
      // ratio of non-zero pixels in the filled region
      double r = (double)countNonZero(maskROI)/(rect.width*rect.height);

      if (r > .45 /* assume at least 45% of the area is filled if it contains text */
          && 
          (rect.height > 8 && rect.width > 8) /* constraints on region size */
          /* these two conditions alone are not very robust. better to use something 
             like the number of significant peaks in a horizontal projection as a third condition */
          )
        {
          rectangle(rgb, rect, Scalar(0, 255, 0), 2);
          vRects.push_back(rect);
        }
    }
  imwrite(OUTPUT_FOLDER_PATH + string("rgb.jpg"), rgb);
  imwrite(OUTPUT_FOLDER_PATH + string("small.jpg"), small);
  imwrite(OUTPUT_FOLDER_PATH + string("large.jpg"), large);
  imwrite(OUTPUT_FOLDER_PATH + string("grad.jpg"), grad);
  imwrite(OUTPUT_FOLDER_PATH + string("bw.jpg"), bw);
  imwrite(OUTPUT_FOLDER_PATH + string("connected.jpg"), connected);
  imwrite(OUTPUT_FOLDER_PATH + string("mask.jpg"), mask);
  return vRects;
}

struct ScreenShot{
  ScreenShot(uint x, uint y, uint width, uint height):
    x(x), y(y), width(width), height(height){

    display = XOpenDisplay(nullptr);
    root = DefaultRootWindow(display);

    XGetWindowAttributes(display, root, &window_attributes);
    screen = window_attributes.screen;
    ximg = XShmCreateImage(display, DefaultVisualOfScreen(screen), DefaultDepthOfScreen(screen), ZPixmap, NULL, &shminfo, width, height);

    shminfo.shmid = shmget(IPC_PRIVATE, ximg->bytes_per_line * ximg->height, IPC_CREAT|0777);
shminfo.shmaddr = ximg->data = (char*)shmat(shminfo.shmid, 0, 0);
shminfo.readOnly = False;
if(shminfo.shmid < 0)
   puts("Fatal shminfo error!");;
   Status s1 = XShmAttach(display, &shminfo);
   printf("XShmAttach() %s\n", s1 ? "success!" : "failure!");

   init = true;
   }

  void operator() (cv::Mat& cv_img){
    if(init)
      init = false;

    XShmGetImage(display, root, ximg, x, y, 0x00ffffff);
    cv_img = cv::Mat(height, width, CV_8UC4, ximg->data);
}

~ScreenShot(){
  if(!init)
    XDestroyImage(ximg);

  XShmDetach(display, &shminfo);
  shmdt(shminfo.shmaddr);
  XCloseDisplay(display);
}

Display* display;
Window root;
XWindowAttributes window_attributes;
Screen* screen;
XImage* ximg;
XShmSegmentInfo shminfo;

int x, y, width, height;

bool init;
};

#if 0
int main() {
  // initilize tesseract OCR engine
  tesseract::TessBaseAPI *myOCR = new tesseract::TessBaseAPI();

  printf("Tesseract-ocr version: %s\n", myOCR->Version());
// printf("Leptonica version: %s\n",
  //        getLeptonicaVersion());

  if (myOCR->Init(NULL, "eng")) {
    fprintf(stderr, "Could not initialize tesseract.\n");
    exit(1);
  }

  //ScreenShot screen(1706, 27+12, 116, 16);
  ScreenShot screen(1706, 4, 116, 16);
  cv::Mat img;

  for(uint i;; ++i){
    double start = clock();

    screen(img);

    if(!(i & 0b111111))
      printf("fps %4.f  spf %.4f\n", FPS(start), 1 / FPS(start));
    break;

  }

  tesseract::PageSegMode pagesegmode = static_cast<tesseract::PageSegMode>(7); // treat the image as a single text line
  myOCR->SetPageSegMode(pagesegmode);

  //using tesseract::TessBaseAPI;
  using tesseract::PageSegMode;

  //myOCR->SetPageSegMode(PageSegMode::PSM_SINGLE_LINE);
  //myOCR->SetVariable("tessedit_char_blacklist", "!?@#$%&*()<>_-+=/:;'\"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
  //myOCR->SetVariable("tessedit_char_whitelist", ".,0123456789");
  //myOCR->SetVariable("classify_bln_numeric_mode", "1");

  img = cv::imread("sample.png");


#if 1
  cv::cvtColor(img, img, CV_BGR2RGBA); 
  myOCR->SetImage(img.data, img.cols, img.rows, 4, 4*img.cols);

  cv::Mat gray = img.clone();
#else // Convert to Gray

  cv::Mat gray;
  cv::cvtColor(img, gray, CV_BGR2GRAY); 
  //myOCR->SetImage(gray.data().asBuffer(),gray.size().width(),gray.size().height(),gray.channels(),gray.size())

  //myOCR->SetImage((uchar*)img.data, img.size().width, img.size().height, img.channels(), img.step1());

  myOCR->SetImage(gray.data, gray.cols, gray.rows, gray.channels(), gray.channels()*gray.cols);

  cv::imwrite("sample_gray.png", gray);
#endif


  //myOCR->TesseractRect( img.data, 1, img.step1(), 0, 0, screen.width, screen.height);
  cv::Rect text1ROI(80, 50, 800, 110);
  myOCR->TesseractRect( img.data, 1, img.step1(), text1ROI.x, text1ROI.y, text1ROI.width, text1ROI.height);


  myOCR->Recognize(0);
  const char *text1 = myOCR->GetUTF8Text();

  // remove "newline"
  std::string t1(text1);
  t1.erase(std::remove(t1.begin(), t1.end(), '\n'), t1.end());

  // print found text
  printf("Found text: \n %s \n", t1.c_str());
  
  //printf("found text1: \n");
  //printf(t1.c_str());
  //printf("\n");
  int confidence = myOCR->MeanTextConf();
  printf("Confidence: %i \n", confidence);
 
  // draw text on original image
  cv::Mat scratch = img.clone();//cv::imread("sample.png");

  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 10;
  int thickness = 2;
  //cv::putText(scratch, t1, cv::Point(0, 0), fontFace, fontScale, cv::Scalar(0, 255, 0), thickness, 8);
  //cv::putText(scratch, t1, cv::Point(0, 0), fontFace, fontScale, cv::Scalar(0, 255, 0), thickness, 8);

  putText(scratch, t1, cv::Point(text1ROI.x, text1ROI.y), fontFace, fontScale, cv::Scalar(0, 255, 0), thickness, 8);

  rectangle(scratch, text1ROI, cv::Scalar(0, 0, 255), 2, 8, 0);


  //cv::imshow("mpv", scratch);
  //cv::imshow("mpv", img);
  //cv::imwrite("sample.png", img);
  cv::imwrite("sample_scratch.png", scratch);

  cv::waitKey(3000);


  delete [] text1;

  // destroy tesseract OCR engine
  myOCR->Clear();
  myOCR->End();
}

#endif

int main(int argc, char* argv[]) {

  // initilize tesseract OCR engine
  tesseract::TessBaseAPI *myOCR =
    new tesseract::TessBaseAPI();

  printf("Tesseract-ocr version: %s\n",
         myOCR->Version());
  // printf("Leptonica version: %s\n",
  //        getLeptonicaVersion());

  if (myOCR->Init(NULL, "eng")) {
    fprintf(stderr, "Could not initialize tesseract.\n");
    exit(1);
  }

  //tesseract::PageSegMode pagesegmode = static_cast<tesseract::PageSegMode>(7); // treat the image as a single text line
  tesseract::PageSegMode pagesegmode = static_cast<tesseract::PageSegMode>(3); // treat the image as a single text line
  myOCR->SetPageSegMode(pagesegmode);
  myOCR->SetVariable("tessedit_enable_dict_correction", "1");
  myOCR->SetVariable("tessedit_char_blacklist", "!?@#$%&*()<>_-+=/:;'\"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
  myOCR->SetVariable("tessedit_char_whitelist", ".,0123456789");
  myOCR->SetVariable("classify_bln_numeric_mode", "0");



  namedWindow("tesseract-opencv", 0);
  //Mat image = imread("sample.png", CV_LOAD_IMAGE_COLOR);
  //Mat image = imread("large.jpg", CV_LOAD_IMAGE_COLOR);

  //ScreenShot screen(1706, 27+19, 116, 16);
  ScreenShot screen(1706, 19, 116, 16);

  Mat image;
  screen(image);
  assert(image.cols !=0 & image.rows != 0);

  // set region of interest (ROI), i.e. regions that contain text
  //Rect text1ROI(, 0, screen.width, screen.height);
  //Rect text2ROI(190, 200, 550, 50);

  //  Rect text1ROI(0, 0, image.cols, image.rows);

  // recognize text
  //  myOCR->TesseractRect( image.data, 1, image.step1(), text1ROI.x, text1ROI.y, text1ROI.width, text1ROI.height);


  vector<Rect> vRects;
  //vRects = findLetterContours(image);
  Rect rect;
  if (!vRects.empty()) {
    //auto rect = vRects[3];
    //assert(vRects.size() == 3);
    //printf("\n----------------%i\n", vRects[3].x);


    //for (auto rect : vRects) {
    //myOCR->TesseractRect( image.data, 1, image.step1(), rect.x, rect.y, rect.width, rect.height);
    //printf(myOCR->GetUTF8Text());
    //}
  } else {
    rect = Rect{0, 0, image.cols, image.rows};
  }


#if 0
    cv::cvtColor(image, image, CV_BGR2RGBA);
    myOCR->SetImage(image.data, image.cols, image.rows, 4, 4*image.cols);

    myOCR->SetImage(gray.data().asBuffer(),gray.size().width(),gray.size().height(),gray.channels(),gray.size());
#else

      cv::cvtColor(image, image, CV_BGR2GRAY); 
    // binarization
    //threshold(image, image, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    myOCR->SetImage((uchar*)image.data, image.size().width, image.size().height, image.channels(), image.step1());

    //myOCR->SetImage(image.data, image.cols, image.rows, image.channels(), image.channels()*image.cols);


#endif

    imwrite("test.png", image);
  myOCR->TesseractRect( image.data, 1, image.step1(), rect.x, rect.y, rect.width, rect.height);

  const char *text1 = myOCR->GetUTF8Text();

  /*
    Scalar color = Scalar(0,255,0);
    for( int i = 0; i < vRects.size(); i++ )
    {
    if(vRects[i].area()<100)continue;
    printf("\n Test\n");
    //drawContours( image, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
    //rectangle( image, vRects[i].tl(), vRects[i].br(),color, 2, 8, 0 );
    }
  */


  // remove "newline"
  string t1(text1);
  t1.erase(std::remove(t1.begin(), t1.end(), '\n'), t1.end());
  // remove whitespace and everything after
  t1.erase(std::remove(t1.begin(), t1.end(), ' '), t1.end());

  // print found text
  printf("found text1: \n");
  printf(t1.c_str());
  printf("\n");


  const char filename[] = "../data/dictionary.txt";
  std::ifstream ifs(filename);
  if (!ifs) {
    std::cerr << "Couldn't open " << filename << " for reading\n";
    return 1;
  }
  MB::bktree<std::string, int, MB::levenshtein_distance> tree;
  std::string word;
  while (ifs >> word) {
    tree.insert(word);
  }
  std::cout << "Inserted " << tree.size() << " words\n";
  std::vector<std::pair<std::string, int> > results;

  std::pair<std::string, int> correct_word("",8);
  const uint limit = 4;
  //t1 = "aj";
  t1 = "31DecenberES";
  std::stringstream ss(extract_ints(t1));
  std::cout << t1 << std::endl;

  if(t1.size()>= 2) {
    tree.find(t1, limit, std::back_inserter(results));
    //for (const auto &it = results.begin(); it != results.end(); ++it) {
    for (const auto &it : results) {
      std::cout << it.first << "(distance " << it.second << ")\n";
      if (it.second <= correct_word.second) {
        correct_word = it;
      }
    }
    if(correct_word.second >= 4) {
      printf("Failed to find correct word.");
      //printf("\nCorrect word: %s\n", correct_word.first);
    } else {
      //      std::cout << correct_word.first << std::endl;
      printf("\nCorrect word: %s\n", correct_word.first.c_str());
    }
    results.clear();

  }


  // draw text on original image
  /*
    Mat scratch = imread("sample.png");

    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 2;
    int thickness = 2;
    putText(scratch, t1, Point(text1ROI.x, text1ROI.y), fontFace, fontScale, Scalar(0, 255, 0), thickness, 8);
    putText(scratch, t2, Point(text2ROI.x, text2ROI.y), fontFace, fontScale, Scalar(0, 255, 0), thickness, 8);

    rectangle(scratch, text1ROI, Scalar(0, 0, 255), 2, 8, 0);
    rectangle(scratch, text2ROI, Scalar(0, 0, 255), 2, 8, 0);

    imshow("tesseract-opencv", scratch);
  */

  //imshow("tesseract-opencv", image);

  waitKey(0);

  delete [] text1;

  // destroy tesseract OCR engine
  myOCR->Clear();
  myOCR->End();

  return 0;
}
