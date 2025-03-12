# Gesture recognition data processing and model training code

This folder contains code to:

* Process data obtained from the Arduino Nano serial monitor (scraped off the consol)
* Train a tensorflow neural network with the data to produce a model for gesture recognition
* Evaluate the model according to a set of basic machine learning metrics of quality
* Embed the model in code ready to program into the Arduino Nano using Arduino IDE

# Graph the input data to make sure it makes sense

```
python3.6.13 graphInputData.py
```

# Processing the raw collected data under ./data/

```
(.env) traiano@Traianos-iMac gesture-recognition % python processFormatData.py 

TensorFlow version = 2.0.0

Processing index 0 for gesture 'punch'.
        There are 15 recordings of the punch gesture.
Processing index 1 for gesture 'flex'.
        There are 14 recordings of the flex gesture.
Processing index 2 for gesture 'sidelift'.
        There are 17 recordings of the sidelift gesture.
Processing index 3 for gesture 'rotcurl'.
        There are 17 recordings of the rotcurl gesture.
Processing index 4 for gesture 'curl'.
        There are 9 recordings of the curl gesture.
Data set parsing and preparation complete.
(.env) traiano@Traianos-iMac gesture-recognition % 
```

# Training the model

```
# use the model to predict the test inputs
predictions = model.predict(inputs_test)

# print the predictions and the expected ouputs
print("predictions =\n", np.round(predictions, decimals=3))
print("actual =\n", outputs_test)

# Plot the predictions along with to the test data
plt.clf()
plt.title('Training data predicted vs actual values')
plt.plot(inputs_test, outputs_test, 'b.', label='Actual')
plt.plot(inputs_test, predictions, 'r.', label='Predicted')
plt.show()
```

 
# Testing the model and coverting to a format for use in Arduino

```
Testing is complete.
2025-03-12 16:57:01.348531: I tensorflow/core/grappler/devices.cc:60] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0 (Note: TensorFlow was not compiled with CUDA support)
2025-03-12 16:57:01.348862: I tensorflow/core/grappler/clusters/single_machine.cc:356] Starting new session
2025-03-12 16:57:01.352149: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:716] Optimization results for grappler item: graph_to_optimize
2025-03-12 16:57:01.352160: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:718]   function_optimizer: function_optimizer did nothing. time = 0.003ms.
2025-03-12 16:57:01.352167: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:718]   function_optimizer: function_optimizer did nothing. time = 0.001ms.
2025-03-12 16:57:01.362400: I tensorflow/core/grappler/devices.cc:60] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0 (Note: TensorFlow was not compiled with CUDA support)
2025-03-12 16:57:01.362542: I tensorflow/core/grappler/clusters/single_machine.cc:356] Starting new session
2025-03-12 16:57:01.366634: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:716] Optimization results for grappler item: graph_to_optimize
2025-03-12 16:57:01.366648: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:718]   constant folding: Graph size after: 17 nodes (-6), 22 edges (-6), time = 2.023ms.
2025-03-12 16:57:01.366654: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:718]   constant folding: Graph size after: 17 nodes (0), 22 edges (0), time = 0.475ms.
Model is 147956 bytes
Header file, model.h, is 776,844 bytes.

Model has been converted to model.h in the output directory.

```


* Check in `./output/` directory:


```
(.env) traiano@Traianos-iMac gesture-recognition % head output/model.h 
const unsigned char model[] = {
  0x1c,0x00,0x00,0x00,0x54,0x46,0x4c,0x33,0x00,0x00,0x12,0x00,
  0x1c,0x00,0x04,0x00,0x08,0x00,0x0c,0x00,0x10,0x00,0x14,0x00,
  0x00,0x00,0x18,0x00,0x12,0x00,0x00,0x00,0x03,0x00,0x00,0x00,
  0x9c,0x41,0x02,0x00,0x18,0x00,0x00,0x00,0x1c,0x00,0x00,0x00,
  0x2c,0x00,0x00,0x00,0x04,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
  0xb4,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0xdc,0x00,0x00,0x00,
  0x0f,0x00,0x00,0x00,0x54,0x4f,0x43,0x4f,0x20,0x43,0x6f,0x6e,
  0x76,0x65,0x72,0x74,0x65,0x64,0x2e,0x00,0x0d,0x00,0x00,0x00,
  0x80,0x00,0x00,0x00,0x74,0x00,0x00,0x00,0x68,0x00,0x00,0x00,


```


# Include the newly generated mode.h, which represents the model itself, in the Arduino code to be programmed into the Arduino nano BLE 33 unit:


```
/*
  IMU Classifier
  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.
  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.
  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.
  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry
  This example code is in the public domain.
*/

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 119;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
namespace {
  tflite::MicroInterpreter* interpreter = nullptr;
  tflite::AllOpsResolver resolver;
  const tflite::Model* tfl_model = nullptr;
}

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "punch",
  "flex"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

// ... rest of the existing code remains unchanged ...
void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // wait for significant motion
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      // read the acceleration data
      IMU.readAcceleration(aX, aY, aZ);

      // sum up the absolutes
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // check if it's above the threshold
      if (aSum >= accelerationThreshold) {
        // reset the sample read count
        samplesRead = 0;
        break;
      }
    }
  }

  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples) {
    // check if new acceleration AND gyroscope data is available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // normalize the IMU data between 0 to 1 and store in the model's
      // input tensor
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples) {
        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
          return;
        }

        // Loop through the output tensor values from the model
        for (int i = 0; i < NUM_GESTURES; i++) {
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i], 6);
        }
        Serial.println();
      }
    }
  }
}

```

* Include model.h as a separate tab in the same arduino sketch so that it can be found during compilation.

```


```


# Program the Arduino

* Compile the sketch
* Upload the code to the Arduino Nano

```
Library Arduino_TensorFlowLite has been declared precompiled:
Precompiled library in "/Users/traiano/Documents/Arduino/libraries/Arduino_TensorFlowLite/src/cortex-m4/fpv4-sp-d16-softfp" not found
Precompiled library in "/Users/traiano/Documents/Arduino/libraries/Arduino_TensorFlowLite/src/cortex-m4" not found
Sketch uses 489536 bytes (49%) of program storage space. Maximum is 983040 bytes.
Global variables use 65712 bytes (25%) of dynamic memory, leaving 196432 bytes for local variables. Maximum is 262144 bytes.
Device       : nRF52840-QIAA
Version      : Arduino Bootloader (SAM-BA extended) 2.0 [Arduino:IKXYZ]
Address      : 0x0
Pages        : 256
Page Size    : 4096 bytes
Total Size   : 1024KB
Planes       : 1
Lock Regions : 0
Locked       : none
Security     : false
Erase flash

Done in 0.001 seconds
Write 490132 bytes to flash (120 pages)

[                              ] 0% (0/120 pages)
[=                             ] 3% (4/120 pages)
[=                             ] 4% (5/120 pages)
[=                             ] 5% (6/120 pages)
[=                             ] 5% (7/120 pages)
[==                            ] 6% (8/120 pages)
[==                            ] 7% (9/120 pages)
[==                            ] 8% (10/120 pages)
[==                            ] 9% (11/120 pages)
[===                           ] 10% (12/120 pages)
[===                           ] 10% (13/120 pages)
[===                           ] 11% (14/120 pages)
[===                           ] 12% (15/120 pages)
[====                          ] 13% (16/120 pages)
[====                          ] 14% (17/120 pages)
[====                          ] 15% (18/120 pages)
[====                          ] 15% (19/120 pages)
[=====                         ] 16% (20/120 pages)
[=====                         ] 17% (21/120 pages)
[=====                         ] 18% (22/120 pages)
[=====                         ] 19% (23/120 pages)
[======                        ] 20% (24/120 pages)
[======                        ] 20% (25/120 pages)
[======                        ] 21% (26/120 pages)
[======                        ] 22% (27/120 pages)
[=======                       ] 23% (28/120 pages)
[=======                       ] 24% (29/120 pages)
[=======                       ] 25% (30/120 pages)
[=======                       ] 25% (31/120 pages)
[========                      ] 26% (32/120 pages)
[========                      ] 27% (33/120 pages)
[========                      ] 28% (34/120 pages)
[========                      ] 29% (35/120 pages)
[=========                     ] 30% (36/120 pages)
[=========                     ] 30% (37/120 pages)
[=========                     ] 31% (38/120 pages)
[=========                     ] 32% (39/120 pages)
[==========                    ] 33% (40/120 pages)
[==========                    ] 34% (41/120 pages)
[==========                    ] 35% (42/120 pages)
[==========                    ] 35% (43/120 pages)
[===========                   ] 36% (44/120 pages)
[===========                   ] 37% (45/120 pages)
[===========                   ] 38% (46/120 pages)
[===========                   ] 39% (47/120 pages)
[============                  ] 40% (48/120 pages)
[============                  ] 40% (49/120 pages)
[============                  ] 41% (50/120 pages)
[============                  ] 42% (51/120 pages)
[=============                 ] 43% (52/120 pages)
[=============                 ] 44% (53/120 pages)
[=============                 ] 45% (54/120 pages)
[=============                 ] 45% (55/120 pages)
[==============                ] 46% (56/120 pages)
[==============                ] 47% (57/120 pages)
[==============                ] 48% (58/120 pages)
[==============                ] 49% (59/120 pages)
[===============               ] 50% (60/120 pages)
[===============               ] 50% (61/120 pages)
[===============               ] 51% (62/120 pages)
[===============               ] 52% (63/120 pages)
[================              ] 53% (64/120 pages)
[================              ] 54% (65/120 pages)
[================              ] 55% (66/120 pages)
[================              ] 55% (67/120 pages)
[=================             ] 56% (68/120 pages)
[=================             ] 57% (69/120 pages)
[=================             ] 58% (70/120 pages)
[=================             ] 59% (71/120 pages)
[==================            ] 60% (72/120 pages)
[==================            ] 60% (73/120 pages)
[==================            ] 61% (74/120 pages)
[==================            ] 62% (75/120 pages)
[===================           ] 63% (76/120 pages)
[===================           ] 64% (77/120 pages)
[===================           ] 65% (78/120 pages)
[===================           ] 65% (79/120 pages)
[====================          ] 66% (80/120 pages)
[====================          ] 67% (81/120 pages)
[====================          ] 68% (82/120 pages)
[====================          ] 69% (83/120 pages)
[=====================         ] 70% (84/120 pages)
[=====================         ] 70% (85/120 pages)
[=====================         ] 71% (86/120 pages)
[=====================         ] 72% (87/120 pages)
[======================        ] 73% (88/120 pages)
[======================        ] 74% (89/120 pages)
[======================        ] 75% (90/120 pages)
[======================        ] 75% (91/120 pages)
[=======================       ] 76% (92/120 pages)
[=======================       ] 77% (93/120 pages)
[=======================       ] 78% (94/120 pages)
[=======================       ] 79% (95/120 pages)
[========================      ] 80% (96/120 pages)
[========================      ] 80% (97/120 pages)
[========================      ] 81% (98/120 pages)
[========================      ] 82% (99/120 pages)
[=========================     ] 83% (100/120 pages)
[=========================     ] 84% (101/120 pages)
[=========================     ] 85% (102/120 pages)
[=========================     ] 85% (103/120 pages)
[==========================    ] 86% (104/120 pages)
[==========================    ] 87% (105/120 pages)
[==========================    ] 88% (106/120 pages)
[==========================    ] 89% (107/120 pages)
[===========================   ] 90% (108/120 pages)
[===========================   ] 90% (109/120 pages)
[===========================   ] 91% (110/120 pages)
[===========================   ] 92% (111/120 pages)
[============================  ] 93% (112/120 pages)
[============================  ] 94% (113/120 pages)
[============================  ] 95% (114/120 pages)
[============================  ] 95% (115/120 pages)
[============================= ] 96% (116/120 pages)
[============================= ] 97% (117/120 pages)
[============================= ] 98% (118/120 pages)
[============================= ] 99% (119/120 pages)
[==============================] 100% (120/120 pages)
Done in 19.166 seconds

```
