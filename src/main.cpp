// g++ screena.cpp -o screena -lX11 -lXext -Ofast -mfpmath=both -march=native -m64 -funroll-loops -mavx2 `pkg-config opencv --cflags --libs` && ./screena

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <X11/extensions/XShm.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <opencv2/opencv.hpp>  // This includes most headers!

#include <cstdint>
#include <cstring>
#include <vector>

#include <time.h>
#define FPS(start) (CLOCKS_PER_SEC / (clock()-start))


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

    XShmGetImage(display, root, ximg, 0, 0, 0x00ffffff);
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


int main(){
  ScreenShot screen(0, 0, 192, 108);
  cv::Mat img;

  for(uint i;; ++i){
    double start = clock();

    screen(img);

    if(!(i & 0b111111))
      printf("fps %4.f  spf %.4f\n", FPS(start), 1 / FPS(start));
    break;

  }

  cv::imshow("img", img);
  cv::waitKey(0);
}

