/*
 * Camera Functions
 *
 * Squirll or Bird
 * Joshua Mehlman
 * Engr 859, Spring 2024
 *
 */



#ifndef _SQB_CAMERA_
#define _SQB_CAMERA_


#include <nuttx/video/video.h>
// Can't init FULLHD ...
//#define JPG_WIDTH  VIDEO_HSIZE_3M
//#define JPG_HEIGHT VIDEO_VSIZE_3M
//#define JPG_WIDTH  VIDEO_HSIZE_HD
//#define JPG_HEIGHT VIDEO_VSIZE_HD
#define JPG_WIDTH  VIDEO_HSIZE_QVGA
#define JPG_HEIGHT VIDEO_VSIZE_QVGA
//#define JPG_WIDTH  24
//#define JPG_HEIGHT 24

// From DSC
const char *initialize_cameractrl(int *fd); // From DSC
int set_ext_ctrls(int fd, uint16_t ctl_cls, uint16_t cid, int value);
unsigned char *camera_framebuffer(int *sz);
int take_pictureimage(int fd, unsigned char **buf, int sz, int w, int h);

int get_previewimage(int fd, unsigned char **buf);
int release_previewimage(int fd, unsigned char *buf);
//int takePicture(int fd, uint8_t **img);
//int takePicture(unsigned char *imageData, int bufferSize);

//void setCameraRez(void);
int initCamera(int *fd);
void closeCamera(int fd);


#endif