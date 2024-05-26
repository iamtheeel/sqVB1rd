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
 * this stuff is worth it, you can buy me a beer in return.   Joshua Mehlman
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


// Main Board Pins
const uint8_t safeMode_pin = 1; //D1 (UART2_TX) Safe Boot

// Extension Board pins

// Built in Leds
const uint8_t emptyLED_pin = LED0;    // 0x40, 64
const uint8_t birdLED_pin = LED1;      // 0x41, 65
const uint8_t squirrelLED_pin = LED2;  // 0x42, 66
const uint8_t heartBeatLED_pin = LED3; // 0x43, 67


/***      saveBoot       ***/
bool safeBoot = false;
unsigned long videoDelay_ms = 0;

/***    Serial    ***/
//#define BAUDRATE (921600)
#define BAUDRATE (1000000)


/***    Camera    ***/
#define JPGQUAL (100)
// Still
//const int iWidth = 96, iHeight = 96; // 
//const int iWidth = CAM_IMGSIZE_QQVGA_H, iHeight = CAM_IMGSIZE_QQVGA_V; // Works
const int iWidth = CAM_IMGSIZE_QVGA_H, iHeight = CAM_IMGSIZE_QVGA_V; //
//const int iWidth = CAM_IMGSIZE_VGA_H, iHeight = CAM_IMGSIZE_VGA_V; // 
//const int iWidth = CAM_IMGSIZE_HD_H, iHeight = CAM_IMGSIZE_HD_V; // 

//const int iWidth = CAM_IMGSIZE_QUADVGA_H, iHeight = CAM_IMGSIZE_QUADVGA_V; // Max with jpeg, Errored when I have the model going
//const int iWidth = CAM_IMGSIZE_FULLHD_H, iHeight = CAM_IMGSIZE_FULLHD_V; // Memory Error, even with 1.5M

// Video
//const int vWidth = 96, vHeight = 96; // The smallest we can do
//const int vWidth = CAM_IMGSIZE_QQVGA_H, vHeight = CAM_IMGSIZE_QQVGA_V; // 
const int vWidth = CAM_IMGSIZE_QVGA_H, vHeight = CAM_IMGSIZE_QVGA_V; // The biggest we can do 
//const int vWidth = CAM_IMGSIZE_VGA_H, vHeight = CAM_IMGSIZE_VGA_V; // No joy(even with minimal memory)

/***      File System     ***/
SDClass  theSD;
int take_picture_count = 0;
char* bird = "Bird";
char* none = "Empty";
char* squirrel = "Squirrel";

// Logger
char logFileName[16];


/***      The Model      ***/
//#include "BGR2BGR_leNetV5.h" ///No
//#include "RGB2BGR_leNetV5.h"
#include "RGB2BGRleNetV5_trained.h"


const int nClasses = 3; // Bird, Nothing, Squirrel

// Image Size for ML
const int mlWidth = 96, mlHeight = 96; //

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// arena size used (you need to load to find out) + imgW*imgH*2 ...(was 3, but that is wrong, yes?)
//constexpr int kTensorArenaSize = 350000;
constexpr int kTensorArenaSize = 165840; // My LeNet from HW4 (400kb)

uint8_t tensor_arena[kTensorArenaSize];


void setup() {
  // Safe Mode Pin setups
  pinMode(safeMode_pin, INPUT_PULLUP);     // Safe mode

  pinMode(emptyLED_pin, OUTPUT); //LED_BUILTIN
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
       Serial.println("No SD card Found!");
       delay(500);
    }
    Serial.println("SD card Mounted.");
    // set up file system
    theSD.mkdir("/DCIM/Bird");
    theSD.mkdir("/DCIM/Empty");
    theSD.mkdir("/DCIM/Squirrel");

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
    static tflite::MicroMutableOpResolver<9> resolver; // The number of adds
    resolver.AddMaxPool2D();
    resolver.AddConv2D();
    resolver.AddRelu();
    resolver.AddTranspose();
    resolver.AddPad();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddDequantize();
    resolver.AddQuantize();
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
    Serial.print((String) "Initilising log");
    int logNum = 0;
    do{ 
      sprintf(logFileName, "%i_log.csv", ++logNum);  // Find the next empty file
    }while(theSD.exists(logFileName));
    Serial.println((String) ", file: " + logFileName);
    File myFile = theSD.open(logFileName, FILE_WRITE);
    myFile.write("FileNum, Bird, Empty, Squirrel\n");
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
    err = theCamera.begin(1, CAM_VIDEO_FPS_5, vWidth, vHeight, CAM_IMAGE_PIX_FMT_YUV422); // Must be YUV422 for clip and resize to work
    //err = theCamera.begin(1, CAM_VIDEO_FPS_5, vWidth, vHeight, CAM_IMAGE_PIX_FMT_RGB565); // Settings we want for the ML Network
    if (err != CAM_ERR_SUCCESS){printError(err);}
    
    // Which camera we got?
    int sensor = theCamera.getDeviceType(); // Begin before asking
    Serial.println((String)"Sensor: " + sensor); // sensor = 1
    
    
    // Auto white balance configuration //
    Serial.println("Set Daylight white balance parameter");
    err = theCamera.setAutoWhiteBalanceMode(CAM_WHITE_BALANCE_DAYLIGHT);
    if (err != CAM_ERR_SUCCESS){printError(err);}
    /*
    //ISO
    Serial.println("Set Auto ISO");
    err = theCamera.setAutoISOSensitivity(true);
    if (err != CAM_ERR_SUCCESS){printError(err);}
    Serial.println("Set Auto Exposure");
    err = theCamera.setAutoExposure(true);
    if (err != CAM_ERR_SUCCESS){printError(err);}
    */
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

void loop() {}  // Everything is callback

void heartBeat(int hbPin)
{
  static bool pinState = 0;

  digitalWrite(hbPin, pinState);
  pinState = !pinState;
}

void setResultsLED(int8_t detect)
{
  // Set everybody off
  digitalWrite(  emptyLED_pin, false);
  digitalWrite(    birdLED_pin, false);
  digitalWrite(squirrelLED_pin, false);

  // Set our detect on
  // The pins are all in a row starting with "None"i
  if(detect > 0) {
    digitalWrite(emptyLED_pin + detect, true);
  }
}

int saveStill(CamImage img, int8_t imageTag)
{
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
    char filename[30] = {0}; // Max len is 21 char
    char birdFilename[30] = {0}; // Max len is 21 char
    char noneFilename[30] = {0}; // Max len is 21 char
    char squiFilename[30] = {0}; // Max len is 21 char
    if(take_picture_count <= 0) // Find the save number
    {
      Serial.println((String) "First Image on boot, find file count");
      do{
        take_picture_count++;
        sprintf(filename,     "/DCIM/%05d.JPG",             take_picture_count);
        sprintf(birdFilename, "/DCIM/%s/%05d.JPG", bird,    take_picture_count);
        sprintf(noneFilename, "/DCIM/%s/%05d.JPG", none,    take_picture_count);
        sprintf(squiFilename, "/DCIM/%s/%05d.JPG", squirrel,take_picture_count);
        //Serial.println((String) "trying " + filename);
      } while(theSD.exists(filename) || theSD.exists(birdFilename) || theSD.exists(noneFilename)|| theSD.exists(squiFilename));
    } 

    switch(imageTag)
    {
      case 0: sprintf(filename, "/DCIM/%s/%05d.JPG", bird, take_picture_count++); break;
      case 1: sprintf(filename, "/DCIM/%s/%05d.JPG", none, take_picture_count++); break;
      case 2: sprintf(filename, "/DCIM/%s/%05d.JPG", squirrel, take_picture_count++); break;
      default:sprintf(filename, "/DCIM/%05d.JPG", take_picture_count++);         // Most will be empty
    }
    
    Serial.println((String) "Save taken picture as " + filename);
    
    File myFile = theSD.open(filename, FILE_WRITE);
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
//#define STREAMRGB
#define SAVEJPG
#define DOINFER
//#define SHOWFULLRGB


    // ********  Take Pix before anything so we have it ready to save ************//
#ifdef SAVEJPG
    //Serial.println((String)"Take a still");
    CamImage stillImg =  theCamera.takePicture();
#endif

  /* Check the img instance is available or not. */
  if (img.isAvailable()) // Turn the results off when thinking about it
  {
    int imageNumber = -1;
    CamErr err;

#ifdef DOINFER
    //Don't fall behind.
    unsigned long lastCamLoop_ms = millis() - videoDelay_ms;
    if((lastCamLoop_ms < 25000) && (videoDelay_ms > 0)){return;}
    videoDelay_ms = millis();
#endif

    digitalWrite(heartBeatLED_pin, true); // Heart Beat when we are thinking
    setResultsLED(-1);

    
    // Resize (and shape?) The image. Lets see if we can trim it.
    //Serial.println((String)"Resize the image");
    CamImage reSizedImg; // New, smaller image

    // 320x240, 96x96
    // (320 - 96)/2 = 112
    // (240 - 96)/2 = 72
    int lefttop_x = (vWidth - mlWidth)/2;   // Center   /**< [en] Left top X coodinate in original image for clipping. <BR> [ja] 元画像に対して、クリップする左上のX座標 */
    int lefttop_y = (vHeight - mlHeight)/2; // Center   /**< [en] Left top Y coodinate in original image for clipping. <BR> [ja] 元画像に対して、クリップする左上のY座標 */
    int rightbottom_x = mlWidth + lefttop_x -1;         /**< [en] Right bottom X coodinate in original image for clipping. <BR> [ja] 元画像に対して、クリップする左上のX座標 */
    int rightbottom_y = mlHeight + lefttop_y -1;        /**< [en] Right bottom Y coodinate in original image for clipping. <BR> [ja] 元画像に対して、クリップする左上のY座標 */
    //Serial.println((String)"Crop Start (x, y): " + lefttop_x + ", " + lefttop_y);
    //Serial.println((String)"Crop End (x, y): " + rightbottom_x + ", " + rightbottom_y);
    err = img.clipAndResizeImageByHW(reSizedImg, lefttop_x, lefttop_y, rightbottom_x, rightbottom_y, mlWidth, mlHeight); // Resize must be in CAM_IMAGE_PIX_FMT_YUV422
    if (err != CAM_ERR_SUCCESS){printError(err);} // Image size must end up one of our good ones.
    //Serial.println((String)"Can we convert to to: CAM_IMAGE_PIX_FMT_JPG"); //No
    
#ifdef SHOWFULLRGB
    err =  img.convertPixFormat(CAM_IMAGE_PIX_FMT_RGB565);
    if (err != CAM_ERR_SUCCESS){printError(err);}
    uint8_t* img_buffer = img.getImgBuff();
    int imageSize = img.getImgSize();
#else
    err =  reSizedImg.convertPixFormat(CAM_IMAGE_PIX_FMT_RGB565);
    if (err != CAM_ERR_SUCCESS){printError(err);}
    uint8_t* img_buffer = reSizedImg.getImgBuff();
    int imageSize = reSizedImg.getImgSize();
#endif
    


#ifdef DOINFER
    // tensorflow inference code
    // Expecting CAM_IMAGE_PIX_FMT_RGB565
    // Send the camera buffer to the input stream
    // From spresense_tf_mnist
    //Serial.println((String)"Put image in memory: " + mlWidth + "x"+ mlHeight );
    for (int i = 0; i < mlWidth * mlHeight * 2; ++i) 
    {
        //Serial.print((String)img_buffer[i] + ", ");
        //if(i%96 ==0){Serial.println();}
      //input->data.f[i] = ((float)(img_buffer[i]));
      input->data.f[i] = ((float)(img_buffer[i])/255);
      //input->data.uint8[i] = img_buffer[i]; // model exported with uint8
    }
    //Serial.println();

    Serial.println((String)"Do inference");
    unsigned long infStart_ms = millis();
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {Serial.println("Invoke failed");return;}
    unsigned long inferenceTime_ms = millis() - infStart_ms;
    Serial.println((String)"Inference time (ms): " + inferenceTime_ms);
#endif

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

    digitalWrite(heartBeatLED_pin, false); // the on time is the save/strem time.. and the 5Hz
    // ********  Echo results  ************//
    static int biCount = 0;
    static int nbCount = 0;
    static int sqCount = 0;
    Serial.print((String)"Bird, None, Squirrel: [" + 
                          output->data.f[0] + ", " + 
                          output->data.f[1] + ", " +
                          output->data.f[2] + "]" ); 
         if(maxIndex == 2){Serial.print((String)", Detected: Squirrel: " + maxValue); sqCount++;}
    else if(maxIndex == 0){Serial.print((String)", Detected: Bird: "     + maxValue); biCount++;}
    else                  {Serial.print((String)", Detected: Empty: "   + maxValue); nbCount++;}
    Serial.println((String)" | Counts: " + biCount + ", " + nbCount + ", " + sqCount);

    // ********  Set LED Status  ************//
    setResultsLED(maxIndex);

    // ********  Save the JPG that we captured just before inference  ************//
#ifdef SAVEJPG
      // Get still before inferance, but save after
      unsigned long fileSave_ms = millis();
      imageNumber = saveStill(stillImg, -1) -1; // -1 to save all the images in one dir
      //imageNumber = saveStill(stillImg, maxIndex); // we increment on the save
      unsigned long fileSaveTime_ms = millis() - fileSave_ms;
      Serial.println((String)"FileSave time (ms): " + fileSaveTime_ms);
#endif

#ifdef STREAMRGB
    // ********  Stream Bitmap Image  ************//


    //Serial.println((String)"SIZE:" + imageSize);
    //Serial.println((String)"START");
    
    Serial.print("A");
    Serial.print("3");
    Serial.print("3");

    Serial.write(img_buffer, imageSize);
    //Serial.write(reSizedImg.getImgBuff(), imageSize);
#endif


    // Only take the picture if we have something.
    //if(maxIndex != 1) // We will also add a max threshold
    //{}

    // We will move the inside the if maxIndex
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