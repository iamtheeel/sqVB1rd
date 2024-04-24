/*
 * Joshua Mehlman
 * ENGR 859 Spring 2024
 * Term Project
 *
 * Squirrl Or Bird Detector
 */

/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42, phk@FreeBSD.ORG):
 * <iamtheeel> wrote this file.  As long as you retain this notice you
 * can do whatever you want with this stuff. If we meet some day, and you think
 * this stuff is worth it, you can buy me a beer in return.   MRM
 * ----------------------------------------------------------------------------
 */
// Standard Librarys
#include <stdio.h>  // sprintf

// From Sony Spresense
#include <Camera.h>
#include <SDHCI.h> // SD Card

// TensorFlow Light
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Self
#include "sqbCamera.h"

// Main Board Pins
const uint8_t safeMode_pin = 1; //D1 (UART2_TX) Safe Boot

// Extension Board pins

// Built in Leds
const uint8_t heartBeat_pin = LED0;
// LED1
// LED2
// LED3


/*      Timing       */
// Heartbeat
int runFreq = 250; // Hz
unsigned long delayTime_us = 1e6/runFreq;
bool safeBoot = false;

unsigned long miliSecTaskClock = 0;
unsigned long uSSystemTaskClock = 0;
byte taskClockCycles25Hz = 0, taskClockCycles10Hz = 0, taskClockCycles5Hz, taskClockCycles1Hz = 0;

/*    Serial    */
#define BAUDRATE (921600)


/*    Camera    */
const int vWidth = CAM_IMGSIZE_QVGA_H, vHeight = CAM_IMGSIZE_QVGA_V; // Works
//const int vWidth = CAM_IMGSIZE_VGA_H, vHeight = CAM_IMGSIZE_VGA_H; // Does not like
//const int iWidth = CAM_IMGSIZE_VGA_H, iHeight = CAM_IMGSIZE_VGA_V; // 
const int iWidth = CAM_IMGSIZE_QUADVGA_H, iHeight = CAM_IMGSIZE_QUADVGA_V; // Max with jpeg
//const int iWidth = CAM_IMGSIZE_FULLHD_H, iHeight = CAM_IMGSIZE_FULLHD_V; // Memory Error
#define JPGQUAL (90)


/*      File System     */
SDClass  theSD;
int take_picture_count = 0;


/*   For Streaming to serial */
//JPEGENC jpgenc;

void setup() {
  // Safe Mode Pin setups
  pinMode(safeMode_pin, INPUT_PULLUP);     // Safe mode
  pinMode(heartBeat_pin, OUTPUT); //LED_BUILTIN

  // Serial Port (USB)
  //Serial.begin(115200);
  Serial.begin(BAUDRATE);
  while (!Serial) {;}  // wait for serial port to connect. Needed for native USB port only 

  safeBoot = !digitalRead(safeMode_pin);
  if(safeBoot)
  {
    // Safe moode bootup
    Serial.println((String) "Start Up: Safe Mode");
  }
  else
  {
    // Main Setups
    Serial.println((String) "Start Up: Normal Mode");

    // SD Card
      /* Initialize SD */
    while (!theSD.begin()) 
    {
       /* wait until SD card is mounted. */
       Serial.println("Insert SD card.");
       delay(500);
    }
    Serial.println("SD card Mounted.");

    // Jpeg setup
    

    // Camera setup // Move to seperate file
    CamErr err;
    //CameraClass theCamera(); it's already there and named "theCamera"
    
    Serial.println("Prepare camera");

    // From Sony camera example
    Serial.println((String)"Set video format: w =" + vWidth + ", h = " + vHeight); // add the FPS
    // Video streem to the ML: 5 FPS is as slow as we can go
    //err = theCamera.begin(); //begin() without parameters means that number of buffers = 1, 30FPS, QVGA, YUV 4:2:2 format
    err = theCamera.begin(1, CAM_VIDEO_FPS_5, vWidth, vHeight, CAM_IMAGE_PIX_FMT_RGB565, 7); // Init video stream to tiny
    //err = theCamera.begin(1, CAM_VIDEO_FPS_5, vWidth, vHeight, CAM_IMAGE_PIX_FMT_YUV422, 7); // Init video stream to tiny
    if (err != CAM_ERR_SUCCESS)    {      printError(err);    }
    

    int sensor = theCamera.getDeviceType(); // Begin before asking
    Serial.println((String)"Sensor: " + sensor); // sensor = 1
    
    Serial.println("Start streaming"); // Not interested in video
    err = theCamera.startStreaming(true, CamCB);
    if (err != CAM_ERR_SUCCESS){printError(err);}
    
    // Auto white balance configuration //
    Serial.println("Set Auto white balance parameter");
    err = theCamera.setAutoWhiteBalanceMode(CAM_WHITE_BALANCE_DAYLIGHT);
    if (err != CAM_ERR_SUCCESS){printError(err);}

    // Still picture for save and to send to serial
    // Dumps core if this has higher res than the video?
    //err = theCamera.create_stillbuff(iWidth,iHeight,CAM_IMAGE_PIX_FMT_YUV422, 7);
    Serial.println((String)"Set still picture format: w =" + iWidth + ", h = " + iHeight);
    //err = theCamera.setStillPictureImageFormat(iWidth,iHeight, CAM_IMAGE_PIX_FMT_YUV422); // Convert doees not like CAM_IMAGE_PIX_FMT_RGB565 --> CAM_IMAGE_PIX_FMT_JPG
    //err = theCamera.setStillPictureImageFormat(iWidth, iHeight, CAM_IMAGE_PIX_FMT_RGB565); // TF Wants RGB565
    err = theCamera.setStillPictureImageFormat(iWidth, iHeight, CAM_IMAGE_PIX_FMT_JPG);
    if (err != CAM_ERR_SUCCESS){printError(err);}
    Serial.println((String)"Set JPEG Quality: " + JPGQUAL);
    err = theCamera.setJPEGQuality(JPGQUAL); // 95 Too much, 90 ok, : At QuadVGA
    if (err != CAM_ERR_SUCCESS){printError(err);}
  }
}



void loop() {
  unsigned long miliSec = millis();
  unsigned long microSec = micros();

  if (microSec - uSSystemTaskClock >= delayTime_us) //Fast Settable Clock Run at runFreq Hz
  {
    uSSystemTaskClock = microSec;
    /*
     *  Main loop
     *  Set runFreq = 200; // Hz
     *  to the desired run frequency
     */ 

  }

if(miliSec - miliSecTaskClock >=10) // 100Hz loop
  {
    miliSecTaskClock = millis();
    // *** 100Hz tasks go here


  
    if(taskClockCycles25Hz == 4) // 25Hz
    {
      taskClockCycles25Hz = 0;

      if(safeBoot) {heartBeat(heartBeat_pin);}
    }
    if (taskClockCycles10Hz == 10) // 10Hz 
    {
      taskClockCycles10Hz = 0;
    }
    if (taskClockCycles5Hz == 20) // 5Hz 
    {
      taskClockCycles5Hz = 0;

      if(!safeBoot) {heartBeat(heartBeat_pin);}
    }

    if (taskClockCycles1Hz == 100) // 1Hz 
    {
      taskClockCycles1Hz = 0;

      //static int foo = 0;
      //if(!safeBoot) {Serial.println((String) "High Mom: " + foo++);}

      //CamImage img =  theCamera.takePicture();
      getStill();
    }

    taskClockCycles1Hz++;
    taskClockCycles5Hz++;
    taskClockCycles10Hz++;
    taskClockCycles25Hz++;
  }
}

void heartBeat(int hbPin)
{
  static bool pinState;

  digitalWrite(hbPin, pinState);
  pinState = !pinState;
}


void printError(enum CamErr err)
{
  Serial.print("Error: ");
  switch (err)
  {
    case CAM_ERR_NO_DEVICE:
      Serial.println("No Device");
      break;
    case CAM_ERR_ILLEGAL_DEVERR:
      Serial.println("Illegal device error");
      break;
    case CAM_ERR_ALREADY_INITIALIZED:
      Serial.println("Already initialized");
      break;
    case CAM_ERR_NOT_INITIALIZED:
      Serial.println("Not initialized");
      break;
    case CAM_ERR_NOT_STILL_INITIALIZED:
      Serial.println("Still picture not initialized");
      break;
    case CAM_ERR_CANT_CREATE_THREAD:
      Serial.println("Failed to create thread");
      break;
    case CAM_ERR_INVALID_PARAM:
      Serial.println("Invalid parameter");
      break;
    case CAM_ERR_NO_MEMORY:
      Serial.println("No memory");
      break;
    case CAM_ERR_USR_INUSED:
      Serial.println("Buffer already in use");
      break;
    case CAM_ERR_NOT_PERMITTED:
      Serial.println("Operation not permitted");
      break;
    default:
      break;
  }
}

void getStill()
{
  CamErr err;
  CamImage img =  theCamera.takePicture();

  /* Check the img instance is available or not. */
  if (img.isAvailable())
  {
        // ********  Send To JPEG to the Serial Port  ************//
    //iSize = (int) img.getImgBuffSize();
    int imageSize = img.getImgSize();

    //Serial.println((String)"iSize size: " + iSize);
    Serial.println((String)"SIZE:" + imageSize);
    Serial.println((String)"START");
    //Serial.write(img.getImgBuff(), imageSize);


    // ********  Save The File  ************//
    // TODO: Find the highest number already there so we don't overwrite see: cartUtils.c.. makes a NUTX call :(
    // Save the file number in FLASH
    char filename[16] = {0};
    do{ 
      sprintf(filename, "DCIM/%05d.JPG", take_picture_count++);    //DCIM/
      //Serial.println((String) "Trying: " + filename);
    }while(theSD.exists(filename));

    Serial.println((String) "Save taken picture as " + filename);
    //theSD.remove(filename); // Remove any file with that name in case there is a cockup
    File myFile = theSD.open(filename, FILE_WRITE);
    Serial.println((String) "writing img");
    myFile.write(img.getImgBuff(), img.getImgSize());
    myFile.close();
    take_picture_count++;
  }
  else
  {
    /*  From SONY
     * The size of a picture may exceed the allocated memory size.
     * Then, allocate the larger memory size and/or decrease the size of a picture.
     * [How to allocate the larger memory]
     * - Decrease jpgbufsize_divisor specified by setStillPictureImageFormat()
     * - Increase the Memory size from Arduino IDE tools Menu
     * [How to decrease the size of a picture]
     * - Decrease the JPEG quality by setJPEGQuality()
     */
    Serial.println("Failed to take picture");
  }
}

void CamCB(CamImage img)
{
  CamErr err;

  /* Check the img instance is available or not. */
  if (img.isAvailable())
  {
    
    int i, iMCUCount, rc, iDataSize, iSize;
    uint8_t *pBuffer;
    uint8_t* img_buffer = img.getImgBuff();

    // tensorflow inference code
/*
    for (int i = 0; i < iWidth * iHeight * 2; ++i) {
      input->data.f[i] = (float)(img_buffer[i]);
    }
*/
/*
    //    Serial.println("Do inference");
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed");
      return;
    }
    int maxIndex = 0; // Variable to store the index of the highest value
    float maxValue = output->data.f[0]; // Assume the first value is the highest
 */

/*
    for (int n = 0; n < 4; ++n) {
      float value = output->data.f[n];

      if (value > maxValue) {
        maxValue = value;
        maxIndex = n;
      }
    }
*/
//    Serial.println("output:" + String(maxIndex));


    
     // ********  Save A Log  ************//


  }
  else
  {
    /*  From SONY
     * The size of a picture may exceed the allocated memory size.
     * Then, allocate the larger memory size and/or decrease the size of a picture.
     * [How to allocate the larger memory]
     * - Decrease jpgbufsize_divisor specified by setStillPictureImageFormat()
     * - Increase the Memory size from Arduino IDE tools Menu
     * [How to decrease the size of a picture]
     * - Decrease the JPEG quality by setJPEGQuality()
     */
    Serial.println("Failed to get video stream");
  }
}