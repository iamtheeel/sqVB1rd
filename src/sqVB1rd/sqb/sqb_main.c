/*
 * Squirll or Bird
 * Joshua Mehlman
 * Engr 859, Spring 2024
 *
 */


#include <stdio.h>
#include <string.h>
#include <unistd.h> // nutx

#include <nuttx/config.h>

#include "cardUtils.h" // Data card and file handeling
#include "camera.h"


#ifdef CONFIG_FS_AUTOMOUNTER_DRIVER
#  include <signal.h>
#  include <fcntl.h>
#  include <sys/ioctl.h>
#  include <nuttx/fs/ioctl.h>
#  include <nuttx/fs/automount.h>
#endif


struct app_param_t
{ // From dsc example
  int vfd;
  unsigned char *fb; // the buffer
  int fb_size;
  //const char *sensorname;
  //int app_mode;
  volatile bool is_card_inserted;
  int nocard_fcnt;
};
static struct app_param_t g_appinst;

//**** For mem card (from DSC)
#define SDCARD_NOTIFY_SIGNO (17)


//********** for image processing ***********// From TF Lite
#include <arch/board/cxd56_imageproc.h>


#define IMG_WIDTH_SAMPLE_REQUIRED     (96)
#define IMG_HEIGHT_SAMPLE_REQUIRED    (96)
#define IMG_DATASIZE_SAMPLE_REQUIRED  \
  (IMG_WIDTH_SAMPLE_REQUIRED*IMG_HEIGHT_SAMPLE_REQUIRED*2)

const imageproc_rect_t ml_clip_rect = {
  .x1 = (JPG_WIDTH  / 2) - IMG_WIDTH_SAMPLE_REQUIRED,
  .y1 = (JPG_HEIGHT / 2) - IMG_HEIGHT_SAMPLE_REQUIRED,
  .x2 = (JPG_WIDTH  / 2) + IMG_WIDTH_SAMPLE_REQUIRED  -1,
  .y2 = (JPG_HEIGHT / 2) + IMG_HEIGHT_SAMPLE_REQUIRED -1
};
static uint8_t *clip_mem;
static uint8_t *convert_mem;


//********** for TF Lite ***********// 
#include <tflmrt/runtime.h>

#define TFLM_TENSOR_ARENA_SIZE  (40 * 1024)

extern unsigned char model_tflite[];
tflm_config_t config = { .cpu_num = 1 };

//********** Main ***********// 
int main(int argc, char *argv[])
{
  int ret = 0;
  int saveNum = -1;
  /* ----       For TFLM     -----*/
  tflm_runtime_t rt;

  printf("Init: \n");
  /* ----       Init Memory Card Status    -----*/
  // TODO: Put this in cardUtils
#ifdef CONFIG_FS_AUTOMOUNTER_DRIVER
  // Init card from DSC example
  int fd;
  sigset_t set;
  struct automount_notify_s notify;

  printf("Init SDCARD\n");
  sigemptyset(&set);
  sigaddset(&set, SDCARD_NOTIFY_SIGNO);

  if (sigprocmask(SIG_BLOCK, &set, NULL) < 0)
    {
      printf("sigprocmask failed\n");
      return -1;
    }

  fd = open("/var/mnt/sd0", 0);
  if (fd < 0)
    {
      printf("open automounter node failed\n");
      return -1;
    }

  notify.an_mount = true;
  notify.an_umount = true;
  notify.an_event.sigev_notify = SIGEV_SIGNAL;
  notify.an_event.sigev_signo = SDCARD_NOTIFY_SIGNO;

  if (ioctl(fd, FIOC_NOTIFY, (uintptr_t)&notify) < 0)
    {
      printf("ioctl automounter node failed\n");
      ret = -1;
      goto errout0;
    }
#endif

  g_appinst.is_card_inserted = file_initialize();
  printf("memoryCard? %i\n", g_appinst.is_card_inserted);
  //g_appinst.nocard_fcnt = 0;

  /* ----       Take User Arguments    -----*/

  /* ----       Init GPIO    -----*/

  /* ----       Init Camera    -----*/  // Frame buffer is init here
  int camErr = initCamera(&g_appinst.vfd);
  if(camErr !=0)
  {
    printf("Camera Error: %i\n", camErr);
    goto errout1;
  }
  /* ----       Alocate the Frame Buffer    -----*/
  // Camera
  g_appinst.fb = camera_framebuffer(&g_appinst.fb_size);
  if (g_appinst.fb == NULL)
    {
      printf("Couldn't Allocate camera memory: %i\n", g_appinst.fb_size);
      ret = -1;
      goto errout2;
    }



  /* ----       Init Image processing    -----*/
  imageproc_initialize();

  //clip_mem = (unsigned char *)memalign(32, IMG_DATASIZE_SAMPLE_REQUIRED);
  //convert_mem = (unsigned char *)memalign(32, IMG_DATASIZE_SAMPLE_REQUIRED);

  /* ----       Init ML Network    -----*/
  void *network;
  network = (void *) model_tflite;

  printf("Load the TF network\n");
  ret = tflm_initialize(&config);
  if (ret)
  {
    printf("tflm_initialize() failed due to %d", ret);
      goto errout2;
  }

  ret = tflm_runtime_initialize(&rt, network, TFLM_TENSOR_ARENA_SIZE);
  if (ret)
  {
    printf("tflm_runtime_initialize() failed due to %d\n", ret);
    goto errout2;
  }
  printf("ARENA Size: %d\n", tflm_runtime_actual_arenasize(&rt));


  /* ----       Keyboard input    -----*/
  // termios.h?


    printf("take picture \n");
    // Take actual picture
    printf("Saving jpg size (%dx%d)\n", JPG_WIDTH, JPG_HEIGHT );
    int jpg_size = take_pictureimage(g_appinst.vfd, &g_appinst.fb, g_appinst.fb_size, JPG_WIDTH, JPG_HEIGHT);
    //jpg_size = take_pictureimage(g_appinst.vfd, &jpg_buf, g_appinst.fb_size, JPG_WIDTH, JPG_HEIGHT);

    // Adjust size TODO: Functionize... 
    // settings for tflmrt_lenet default:
    // image size = 28x29
    // Color depth 1x uint8_t (8 bit grey)
    // Scale: 
    // STD:
    /*
    ret = imageproc_clip_and_resize(g_appinst.fb, JPG_WIDTH, JPG_HEIGHT,
        clip_mem, IMG_WIDTH_SAMPLE_REQUIRED, IMG_HEIGHT_SAMPLE_REQUIRED,
        16, (imageproc_rect_t *)&ml_clip_rect); // 16:YUV422 
    if(ret !=0)
    {
      printf("Failed to clip and resize image: %i\n", ret);
      goto errout2;
    }
    imageproc_convert_yuv2gray(clip_mem, convert_mem,
              IMG_WIDTH_SAMPLE_REQUIRED, IMG_HEIGHT_SAMPLE_REQUIRED);
    // convert_mem is acutaly a TfLiteTensor... But we are working on that

    */ 

    // Send to ML

    if (jpg_size > 0) //&& if we like the ML results
    {
      saveNum = file_writeimage(g_appinst.fb, (size_t)jpg_size);
      //saveNum = file_writeimage(clip_mem, (size_t)IMG_DATASIZE_SAMPLE_REQUIRED);
      //saveNum = file_writeimage(convert_mem, (size_t)IMG_DATASIZE_SAMPLE_REQUIRED);
    }
    //int frameNum = taking_picture(&img);

    printf("done \n");
  /*
  while(1)
  {
    takePicture(g_appinst.fb);
    printf("foo\n");
    sleep(1);
  }
  */

errout2:
  tflm_runtime_finalize(&rt);
  tflm_finalize();
errout1:
  closeCamera(g_appinst.vfd);
  file_finalize();

#ifdef CONFIG_FS_AUTOMOUNTER_DRIVER
errout0:
  close(fd); // Close up our files
#endif
  return ret;
} // End Main
