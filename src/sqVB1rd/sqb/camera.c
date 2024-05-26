/*
 * Camera Functions
 *
 * Squirll or Bird
 * Joshua Mehlman
 * Engr 859, Spring 2024
 *
 */

#include "camera.h"
#include <arch/board/cxd56_imageproc.h>

#include <stdio.h>
#include <sys/ioctl.h>

#include <fcntl.h>

#include "cardUtils.h"


// ********** from dsc *************** //
#define VIDEO_DEVF "/dev/video"
#define PREVIEW_BUFFER_NUM  (2)
#define STILL_BUFFER_NUM  (1)
static int request_camerabuffs(int fd, enum v4l2_buf_type type)
{
  struct v4l2_requestbuffers req;

  req.type   = type;
  req.memory = V4L2_MEMORY_USERPTR;
  req.count  = type == V4L2_BUF_TYPE_STILL_CAPTURE
                 ? STILL_BUFFER_NUM
                 : PREVIEW_BUFFER_NUM;
  req.mode   = V4L2_BUF_MODE_FIFO;

  return ioctl(fd, VIDIOC_REQBUFS, (unsigned long)&req);
}

static int enqueue_framebuffer(int fd, unsigned char *buf, int buf_size,
                               enum v4l2_buf_type type)
{
  struct v4l2_buffer vbuf;

  vbuf.type = type;
  vbuf.memory = V4L2_MEMORY_USERPTR;
  vbuf.index = 0;
  vbuf.m.userptr = (unsigned long)buf;
  vbuf.length = buf_size;

  return ioctl(fd, VIDIOC_QBUF, (unsigned long)&vbuf);
}

static int start_stillcapture(int fd)
{

  return ioctl(fd, VIDIOC_TAKEPICT_START, 0);
}

static int stop_stillcapture(int fd)
{
  return ioctl(fd, VIDIOC_TAKEPICT_STOP, 0);
}

static int set_cameraformat(int fd, int width, int height, enum v4l2_buf_type type, uint32_t pixfmt)
{
  struct v4l2_format fmt;

  fmt.type                = type;
  fmt.fmt.pix.width       = (uint16_t)width;
  fmt.fmt.pix.height      = (uint16_t)height;
  fmt.fmt.pix.field       = V4L2_FIELD_ANY;
  fmt.fmt.pix.pixelformat = pixfmt;

  return ioctl(fd, VIDIOC_S_FMT, (unsigned long)&fmt);
}
static int dequeue_framebuffer(int fd, unsigned char **buf,
                               enum v4l2_buf_type type)
{
  int ret = -EBUSY;

  struct v4l2_buffer v_buf;

  memset(&v_buf, 0, sizeof(v_buf));
  v_buf.type = type;
  v_buf.memory = V4L2_MEMORY_USERPTR;

  if (ioctl(fd, VIDIOC_DQBUF, (unsigned long)&v_buf) == 0)
  {
      *buf = (unsigned char *)v_buf.m.userptr;
      ret = v_buf.bytesused;
  }

  return ret;
}
static int stop_stream_local(int fd, enum v4l2_buf_type type)
{
  return ioctl(fd, VIDIOC_STREAMOFF, (unsigned long)&type);
}


/****************************************************************************
 * Public Functions
 ****************************************************************************/


int take_pictureimage(int fd, unsigned char **buf, int sz, int w, int h)
{
  int ret;
  static int compresh = 75;

  // Josh woz here
  //ret = ioctl(fd, V4L2_CID_HFLIP, 1); 
  //ret = ioctl(fd, V4L2_CID_VFLIP, 1); 

  ret = stop_stream_local(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE);

  ret = set_ext_ctrls(fd, V4L2_CTRL_CLASS_CAMERA,
                           V4L2_CID_EXPOSURE_METERING,
                           TRUE ? V4L2_EXPOSURE_METERING_CENTER_WEIGHTED : V4L2_EXPOSURE_METERING_AVERAGE);
  ret = set_ext_ctrls(fd, V4L2_CTRL_CLASS_CAMERA, V4L2_CID_ISO_SENSITIVITY_AUTO, TRUE);

  ret = set_ext_ctrls(fd, V4L2_CTRL_CLASS_CAMERA, V4L2_CID_HFLIP_STILL, TRUE);
  ret = set_ext_ctrls(fd, V4L2_CTRL_CLASS_CAMERA, V4L2_CID_VFLIP_STILL, TRUE);

  //I wanted to try out zoom
  //foo = set_ext_ctrls(fd, V4L2_CTRL_CLASS_CAMERA, V4L2_CID_ZOOM_ABSOLUTE, 2);

  // The default compression was makikng corrupt images in high light
  // 9 seems to give shite... 
  // https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/ext-ctrls-jpeg.html
  // Says higher is better from 1 to 100, 0 by default. Is 0 off?
  if(compresh > 100) {compresh = 0;}
  //printf("compresh = %i\n", compresh);
  ret = set_ext_ctrls(fd, V4L2_CTRL_CLASS_JPEG, V4L2_CID_JPEG_COMPRESSION_QUALITY, compresh);
  //compresh += 5;

  request_camerabuffs(fd, V4L2_BUF_TYPE_STILL_CAPTURE);
  set_cameraformat(fd, w, h, V4L2_BUF_TYPE_STILL_CAPTURE, V4L2_PIX_FMT_JPEG);

  ret = enqueue_framebuffer(fd, *buf, sz, V4L2_BUF_TYPE_STILL_CAPTURE);
  if (ret < 0)
    {
      goto takepict_exit;
    }

  ret = start_stillcapture(fd);
  if (ret < 0)
    {
      return ret;
    }

  ret = dequeue_framebuffer(fd, buf, V4L2_BUF_TYPE_STILL_CAPTURE);

takepict_exit:

  stop_stillcapture(fd);

  return ret;
}

static const char *get_imgsensor_name(int fd)
{
  static struct v4l2_capability cap;

  ioctl(fd, VIDIOC_QUERYCAP, (uintptr_t)&cap);

  return (FAR const char *)cap.driver;
}

int set_ext_ctrls(int fd, uint16_t ctl_cls, uint16_t cid, int value)
{
  struct v4l2_ext_controls ctrls;
  struct v4l2_ext_control control;

  control.id = cid;
  control.value = value;

  ctrls.count = 1;
  ctrls.ctrl_class = ctl_cls;
  ctrls.controls = &control;

  return ioctl(fd, VIDIOC_S_EXT_CTRLS, (unsigned long)&ctrls);
}


static int allocate_framebuffer(int width, int height,
                                uint32_t pix, unsigned char **buf)
{
  int fbsize = width * height * 2;
  printf("w = %i, h = %i, fbsize: %i\n", width, height, fbsize);

  // is V4L..JPEG == fbsize/7
  fbsize = pix == V4L2_PIX_FMT_JPEG ? fbsize / 7 : fbsize;
  printf("allocate_framebuffer: size = %i, pix=%li, V4L2_xx_JPEJ = %li\n", fbsize, pix, V4L2_PIX_FMT_JPEG);
  *buf = (unsigned char *)memalign(32, fbsize);
  printf("mem ret:%hhn \n", *buf);

  return *buf != NULL ? fbsize : -ENOMEM;
}


unsigned char *camera_framebuffer(int *sz)
{
  unsigned char *ret;

  /* Allocate biggest size of this usecase (JPEG on FullHD) */

  // width, height, format
  //*sz = allocate_framebuffer(VIDEO_HSIZE_QVGA, VIDEO_VSIZE_QVGA, V4L2_PIX_FMT_JPEG, &ret);  // can not alocate camera memory past 80
  //*sz = allocate_framebuffer(VIDEO_HSIZE_HD, VIDEO_VSIZE_HD, V4L2_PIX_FMT_JPEG, &ret);  // can not alocate camera memory past 80
  //*sz = allocate_framebuffer(VIDEO_HSIZE_QUADVGA, VIDEO_VSIZE_QUADVGA, V4L2_PIX_FMT_JPEG, &ret);  // 2,457,600, ok
  //*sz = allocate_framebuffer(VIDEO_HSIZE_FULLHD, VIDEO_VSIZE_FULLHD, V4L2_PIX_FMT_JPEG, &ret); // 4,147,200 ,  ok
  //*sz = allocate_framebuffer(VIDEO_HSIZE_3M, VIDEO_VSIZE_3M, V4L2_PIX_FMT_JPEG, &ret);      // 6,291,456, not ok
  *sz = allocate_framebuffer(JPG_WIDTH, JPG_HEIGHT, V4L2_PIX_FMT_JPEG, &ret);  // can not alocate camera memory past 80

  return ret;
}

int initCamera(int *fd)
{
    int err = 0;
    FAR const char *sensor;

    printf("Create video device\n");
    *fd = video_initialize(VIDEO_DEVF); // Create the device file
    if(*fd != 0){
      printf("Could not init video device!!\n");
      return *fd;}

    printf("Open video device\n");
    *fd = open(VIDEO_DEVF, 0); // Open the file with No flags
    if(*fd < 0){
      printf("Could not open video device file!!\n");
        video_uninitialize(VIDEO_DEVF); 
        return *fd;
    }

    printf("Get Camera Buffer\n");
    err = request_camerabuffs(*fd, V4L2_BUF_TYPE_STILL_CAPTURE);
    if (err < 0)
    {
      printf("Couldn't Allocate camera memory for still camera\n");
      close(*fd);
      video_uninitialize(VIDEO_DEVF); 
      *fd = -EINVAL;
      return err;
    }

/*
    err = request_camerabuffs(*fd, V4L2_BUF_TYPE_VIDEO_CAPTURE);
    if (err < 0)
    {
      printf("Couldn't Allocate camera memory for videio camera\n");
      close(*fd);
      video_uninitialize(VIDEO_DEVF); 
      *fd = -EINVAL;
      return err;
    }
*/

    sensor = get_imgsensor_name(*fd);
    printf("Camera name: %s\n", sensor);
    // ISX012 = FULLHD
/*
    err = camera_prepare(V4L2_BUF_TYPE_VIDEO_CAPTURE,
                       V4L2_BUF_MODE_RING, V4L2_PIX_FMT_UYVY,
                       VIDEO_HSIZE_QVGA, VIDEO_VSIZE_QVGA);
*/

    return err;
}

int get_previewimage(int fd, unsigned char **buf)
{
  //int foo = set_ext_ctrls(fd, V4L2_CTRL_CLASS_CAMERA, V4L2_CID_HFLIP_STILL, TRUE);
  //foo = set_ext_ctrls(fd, V4L2_CTRL_CLASS_CAMERA, V4L2_CID_VFLIP_STILL, TRUE);
  return dequeue_framebuffer(fd, buf, V4L2_BUF_TYPE_VIDEO_CAPTURE);
}
int release_previewimage(int fd, unsigned char *buf)
{
  int preview_size = VIDEO_HSIZE_QVGA * VIDEO_VSIZE_QVGA * 2;
  return enqueue_framebuffer(fd, buf, preview_size, V4L2_BUF_TYPE_VIDEO_CAPTURE);
}

void closeCamera(int fd)
{
    close(fd);
    video_uninitialize("/dev/video"); 
}