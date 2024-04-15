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

#include "cardUtils.h"

#ifdef CONFIG_FS_AUTOMOUNTER_DRIVER
#  include <signal.h>
#  include <fcntl.h>
#  include <sys/ioctl.h>
#  include <nuttx/fs/ioctl.h>
#  include <nuttx/fs/automount.h>
#endif

struct app_param_t
{ // From dsc example
  //int vfd;
  //unsigned char *fb;
  //int fb_size;
  //int lcd_w;
  //int lcd_h;
  //const char *sensorname;
  //int app_mode;
  volatile bool is_card_inserted;
  int nocard_fcnt;
};
static struct app_param_t g_appinst;

//**** For mem card (from DSC)
#define SDCARD_NOTIFY_SIGNO (17)

int main(int argc, char *argv[])
{
  int ret = 0;

  /* ----       Init Memory Card Status    -----*/
#ifdef CONFIG_FS_AUTOMOUNTER_DRIVER
  // Init card from DSC example
  int fd;
  sigset_t set;
  struct automount_notify_s notify;

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
  g_appinst.nocard_fcnt = 0;


  printf("Rebooting Server is excilit plan!\n");
  //return 0;

//errout1:
  file_finalize();

#ifdef CONFIG_FS_AUTOMOUNTER_DRIVER
errout0:
  close(fd); // Close up our files
#endif
  return ret;
}
