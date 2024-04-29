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
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
//#include "tensorflow/lite/micro/micro_error_reporter.h" // Using err from MIL SPRESENS_TensorFlow Board
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/spresense/debug_log_callback.h"


// Self
//#include "sqbCamera.h"

// Main Board Pins
const uint8_t safeMode_pin = 1; //D1 (UART2_TX) Safe Boot

// Extension Board pins

// Built in Leds
const uint8_t nobodyLED_pin = LED0;    // 0x40, 64
const uint8_t birdLED_pin = LED1;      // 0x41, 65
const uint8_t squirrelLED_pin = LED2;  // 0x42, 66
const uint8_t heartBeatLED_pin = LED3; // 0x43, 67


/***      Timing       ***/
// Heartbeat
int runFreq = 250; // Hz
unsigned long delayTime_us = 1e6/runFreq;
bool safeBoot = false;

unsigned long miliSecTaskClock = 0;
unsigned long uSSystemTaskClock = 0;
byte taskClockCycles25Hz = 0, taskClockCycles10Hz = 0, taskClockCycles5Hz, taskClockCycles1Hz = 0;

//int everyNthImage = 5; // Slow down, try 1Hz

/***    Serial    ***/
#define BAUDRATE (921600)


/***    Camera    ***/
#define JPGQUAL (90)

//const int iWidth = 96, iHeight = 96; // 
//const int iWidth = CAM_IMGSIZE_QQVGA_H, iHeight = CAM_IMGSIZE_QQVGA_V; // Works
//const int iWidth = CAM_IMGSIZE_QVGA_H, iHeight = CAM_IMGSIZE_QVGA_V; //
const int iWidth = CAM_IMGSIZE_VGA_H, iHeight = CAM_IMGSIZE_VGA_V; // 
//const int iWidth = CAM_IMGSIZE_QUADVGA_H, iHeight = CAM_IMGSIZE_QUADVGA_V; // Max with jpeg, Errored when I have the model going
//const int iWidth = CAM_IMGSIZE_FULLHD_H, iHeight = CAM_IMGSIZE_FULLHD_V; // Memory Error, even with 1.5M


/***      File System     ***/
SDClass  theSD;
int take_picture_count = 0;

// Logger
char logFileName[16];


/***      The Model      ***/
//#include "model.h"
#include "leNetV5.h"
const int nClasses = 3; // Nothing, Bird, Squirrel

// Image Size for ML
//const int vWidth = 48, vHeight = 48; // Does not like 48x48
const int vWidth = 96, vHeight = 96; // 
//const int vWidth = CAM_IMGSIZE_QQVGA_H, vHeight = CAM_IMGSIZE_QQVGA_V; // 
//const int vWidth = CAM_IMGSIZE_QVGA_H, vHeight = CAM_IMGSIZE_QVGA_V; // 
//const int vWidth = CAM_IMGSIZE_VGA_H, vHeight = CAM_IMGSIZE_VGA_H; // 

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

//constexpr int kTensorArenaSize = 350000;
constexpr int kTensorArenaSize = 90000; // arena at boot + imgW*imgH*3

uint8_t tensor_arena[kTensorArenaSize];


void setup() {
  // Safe Mode Pin setups
  pinMode(safeMode_pin, INPUT_PULLUP);     // Safe mode

  pinMode(nobodyLED_pin, OUTPUT); //LED_BUILTIN
  pinMode(birdLED_pin, OUTPUT); //LED_BUILTIN
  pinMode(squirrelLED_pin, OUTPUT); //LED_BUILTIN
  pinMode(heartBeatLED_pin, OUTPUT); //LED_BUILTIN

  // Serial Port (USB)
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

    /***      Initialize SD     ***/
    while (!theSD.begin()) 
    {
       //wait until SD card is mounted.
       Serial.println("Insert SD card.");
       delay(500);
    }
    Serial.println("SD card Mounted.");
    

    /***             Model setup  From Spresense_tf_mnist            ***/
    tflite::InitializeTarget();
    Serial.println((String)"Alocate Areana: " + kTensorArenaSize*sizeof(uint8_t));
    memset(tensor_arena, 0, kTensorArenaSize*sizeof(uint8_t));
  

    // Model Error Reporting
    Serial.println((String)"Set up ML Error Reporting");
    tflite::ErrorReporter* error_reporter = nullptr;

    // Map the model into a usable data structure..
    model = tflite::GetModel(leNetV5_tflite);
    //model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      Serial.println("Model provided is schema version " 
                    + String(model->version()) + " not equal "
                    + "to supported version "
                    + String(TFLITE_SCHEMA_VERSION));
      return;
    } else {
      Serial.println("Model version: " + String(model->version()));
    }

    // This pulls in all the operation implementations we need.
     Serial.println((String)"Start resolver: ");
    //static tflite::AllOpsResolver resolver;
    static tflite::MicroMutableOpResolver<8> resolver; // The number of adds
    resolver.AddMaxPool2D();
    resolver.AddConv2D();
    resolver.AddRelu();
    resolver.AddTranspose();
    resolver.AddPad();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddDequantize();
    RegisterDebugLogCallback(debug_log_printf);

    
    // Build an interpreter to run the model with.
    Serial.println((String)"Alocate Interpriter: ");
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);//, error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate memory from the tensor_arena for the model's tensors.
    Serial.println((String)"Alocate Tensors: ");
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      Serial.println("AllocateTensors() failed");
      return;
    } else {
      Serial.println("AllocateTensor() Success");
    }

    size_t used_size = interpreter->arena_used_bytes();
    Serial.println("Arnea used bytes: " + String(used_size));

    input = interpreter->input(0);
    output = interpreter->output(0);

    /***             Logging setup              ***/ //Move to seperate file
    int logNum = 0;
    do{ 
      sprintf(logFileName, "%i_log.csv", ++logNum);  // Find the next empty file
    }while(theSD.exists(logFileName));

    Serial.println((String) "Initilising log file: " + logFileName);
    File myFile = theSD.open(logFileName, FILE_WRITE);
    myFile.write("FileNum, none, b1rd, squirrl\n");
    myFile.close();



    /***             Camera setup              ***/ //Move to seperate file
    // The very last thing in setup is to call startStreaming. This runs the callback
    // Init the camera after the model so we can test the memory
    CamErr err;
    //CameraClass theCamera(); it's already there and named "theCamera"
    Serial.println("Prepare camera");
    // From Sony camera example
    Serial.println((String)"Set video format: w = " + vWidth + ", h = " + vHeight); // add the FPS
    // Video streem to the ML: 5 FPS is as slow as we can go
    // RGB 565 is 2 bytes/pixl (R:HB7-HB3, G:HB2-LB5, B: LB4-LB0)
    err = theCamera.begin(1, CAM_VIDEO_FPS_5, vWidth, vHeight, CAM_IMAGE_PIX_FMT_RGB565); // Settings we want for the ML Network
    if (err != CAM_ERR_SUCCESS){printError(err);}
    
    // Which camera we got?
    int sensor = theCamera.getDeviceType(); // Begin before asking
    Serial.println((String)"Sensor: " + sensor); // sensor = 1
    
    
    // Auto white balance configuration //
    Serial.println("Set Auto white balance parameter");
    err = theCamera.setAutoWhiteBalanceMode(CAM_WHITE_BALANCE_DAYLIGHT);
    if (err != CAM_ERR_SUCCESS){printError(err);}

    //ISO


    // Still picture for save and to send to serial
    Serial.println((String)"Set still picture format: w =" + iWidth + ", h = " + iHeight);
    err = theCamera.setStillPictureImageFormat(iWidth, iHeight, CAM_IMAGE_PIX_FMT_JPG, 1); // JPEG Divisor requests less memory assuming good compresion, 1 = assume full image
    if (err != CAM_ERR_SUCCESS){printError(err);}

    Serial.println((String)"Set JPEG Quality: " + JPGQUAL);
    err = theCamera.setJPEGQuality(JPGQUAL); // 95 Too much, 90 ok, : At QuadVGA
    if (err != CAM_ERR_SUCCESS){printError(err);}
 

    // Start the video and register the callback
    // Do this LAST!
    Serial.println("Start streaming"); 
    err = theCamera.startStreaming(true, CamCB);
    if (err != CAM_ERR_SUCCESS){printError(err);}


    /***             Done With Setup              ***/
    Serial.println((String)"Init Done");
  } // END SafeBoot
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

      if(safeBoot) {heartBeat(heartBeatLED_pin);}
      //else{}
    }
    if (taskClockCycles10Hz == 10) // 10Hz 
    {
      taskClockCycles10Hz = 0;
      //if(!safeBoot) {Serial.println((String) "High Mom: " + foo++);}
    }
    if (taskClockCycles5Hz == 20) // 5Hz 
    {
      taskClockCycles5Hz = 0;

      if(!safeBoot) {
        //heartBeat(heartBeatLED_pin);
      }
      //else {; }
    }

    if (taskClockCycles1Hz == 100) // 1Hz 
    {
      taskClockCycles1Hz = 0;

      if(!safeBoot) {
        //getStill();
      }

      
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

void setResultsLED(uint8_t detect)
{
  // Set everybody off
  digitalWrite(  nobodyLED_pin, false);
  digitalWrite(    birdLED_pin, false);
  digitalWrite(squirrelLED_pin, false);

  // Set our detect on
  // The pins are all in a row starting with "None"
  digitalWrite(nobodyLED_pin + detect, true);
}

int getStill()
{
  CamErr err;
  CamImage img =  theCamera.takePicture();

  /* Check the img instance is available or not. */
  if (img.isAvailable())
  {
    // ********  Send To JPEG to the Serial Port  ************//
    //iSize = (int) img.getImgBuffSize();
    int imageSize = img.getImgSize();
    //Serial.println((String)"SIZE:" + imageSize);
    //Serial.println((String)"START");
    //Serial.write(img.getImgBuff(), imageSize);


    // ********  Save The File  ************//
    // Save 
    char filename[30] = {0};
    do{ 
      sprintf(filename, "DCIM/%05d.JPG", ++take_picture_count);    //DCIM/
      //Serial.println((String) "Trying: " + filename);
    }while(theSD.exists(filename));

    Serial.println((String) "Save taken picture as " + filename);
    //theSD.remove(filename); // Remove any file with that name in case there is a cockup
    File myFile = theSD.open(filename, FILE_WRITE);
    Serial.println((String) "writing img");
    myFile.write(img.getImgBuff(), img.getImgSize());
    myFile.close();
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
    Serial.println("Failed to take picture: Mem Allocate Error?");
  }

  return take_picture_count;
}

void CamCB(CamImage img)
{
  CamErr err;

  /* Check the img instance is available or not. */
  if (img.isAvailable())
  {
    digitalWrite(heartBeatLED_pin, true); // Heart Beat when we are thinking
    //CamImage reSizedImg; // New, smaller image
    //err = img.resizeImageByHW(reSizedImg, 48, 48); //Damned invalide peram, 96x96 seems to be the smallest
    //if (err != CAM_ERR_SUCCESS){printError(err);}

    int i, iMCUCount, rc, iDataSize, iSize;
    uint8_t *pBuffer;
    uint8_t* img_buffer = img.getImgBuff();

    // tensorflow inference code
    // Expecting CAM_IMAGE_PIX_FMT_RGB565
    // Send the camera buffer to the input stream
    // From spresense_tf_mnist
    //Serial.println((String)"Put image in memory: " + vWidth + "x"+ vHeight );
    for (int i = 0; i < vWidth * vHeight * 2; ++i) {
      input->data.f[i] = (float)(img_buffer[i]);
    }


    //Serial.println((String)"Do inference");
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {Serial.println("Invoke failed");return;}

   //Serial.println((String)"Gather Results");
    uint8_t maxIndex = 0;               // Variable to store the index of the highest value
    float maxValue = output->data.f[0]; // The max is our guy
    for (int n = 0; n < nClasses; ++n) {
      float value = output->data.f[n];

      if (value > maxValue) {
        maxValue = value;
        maxIndex = n;
      }
    }
    digitalWrite(heartBeatLED_pin, false);

    // ********  Echo results  ************//
    Serial.print((String)"None, Bird, Squirrel: [" + 
                          output->data.f[0] + ", " + 
                          output->data.f[1] + ", " +
                          output->data.f[2] + "]" ); 

         if(maxIndex == 2){Serial.println((String)", Detected: Squirrel: " + maxValue);}
    else if(maxIndex == 1){Serial.println((String)", Detected: Bird: "     + maxValue);}
    else                  {Serial.println((String)", Detected: Nobody: "   + maxValue);}

    // ********  Set LED Status  ************//
    setResultsLED(maxIndex);

    // ********  Stream Bitmap Image  ************//
    int imageSize = img.getImgSize();
    //Serial.println((String)"SIZE:" + imageSize);
    //Serial.println((String)"START");
    Serial.print("X");
    Serial.write(img.getImgBuff(), imageSize);


    // ********  Take Pix  ************//
    int imageNumber = 0;
    // Only take the picture if we have something.
    if(maxIndex != 0) 
    {
      imageNumber = getStill();
    }

    // ********  Save A Log  ************//
    // Log Contents: Save Number, confidence array
    File logFile = theSD.open(logFileName, FILE_WRITE);
    char logText[80] = {0};
    sprintf(logText, "%i, %f, %f, %f\n", imageNumber, output->data.f[0], output->data.f[1], output->data.f[2]);   
    logFile.write(logText);
    logFile.close();


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

void printError(enum CamErr err)
{
  Serial.print("Error: ");
  switch (err)
  {
    case CAM_ERR_NO_DEVICE:             Serial.println("No Device");                      break;
    case CAM_ERR_ILLEGAL_DEVERR:        Serial.println("Illegal device error");           break;
    case CAM_ERR_ALREADY_INITIALIZED:   Serial.println("Already initialized");            break;
    case CAM_ERR_NOT_INITIALIZED:       Serial.println("Not initialized");                break;
    case CAM_ERR_NOT_STILL_INITIALIZED: Serial.println("Still picture not initialized");  break;
    case CAM_ERR_CANT_CREATE_THREAD:    Serial.println("Failed to create thread");        break;
    case CAM_ERR_INVALID_PARAM:         Serial.println("Invalid parameter");              break;
    case CAM_ERR_NO_MEMORY:             Serial.println("No memory");                      break;
    case CAM_ERR_USR_INUSED:            Serial.println("Buffer already in use");          break;
    case CAM_ERR_NOT_PERMITTED:         Serial.println("Operation not permitted");        break;
    default:                            Serial.println("Unknown Error");                  break;
  }
}

void debug_log_printf(const char* s)
{
  Serial.print("ERROR:");
  Serial.println(s);
}