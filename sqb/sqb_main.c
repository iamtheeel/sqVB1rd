#include <nuttx/config.h>
#include <stdio.h>

#ifdef CONFIG_FS_AUTOMOUNTER_DRIVER
#  include <signal.h>
#  include <fcntl.h>
#  include <sys/ioctl.h>
#  include <nuttx/fs/ioctl.h>
#  include <nuttx/fs/automount.h>
#endif

int main(int argc, char *argv[])
{
  printf("Rebooting Server is excilit plan!\n");
  return 0;
}
