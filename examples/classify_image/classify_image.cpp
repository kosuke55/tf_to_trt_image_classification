/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * Full license terms provided in LICENSE.md file.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char * msg) override
  {
    if (severity != Severity::kINFO)
      std::cout << msg << std::endl;
  }
} gLogger;


/**
 * image_file: path to image
 * plan_file: path of the serialized engine file
 * label_file: file with <class_name> per line
 * input_name: name of the input tensor
 * output_name: name of the output tensor
 */
int main(int argc, char *argv[])
{
  if (argc != 6)
  {
    std::cout << "Usage: classify_image <image_file> <plan_file> <label_file> <input_name> <output_name>";
    return 0;
  }

  std::string imageFilename = argv[1];
  std::string planFilename = argv[2];
  std::string labelFilename = argv[3];
  std::string inputName = argv[4];
  std::string outputName = argv[5];

  /* load the engine */
  std::cout << "Loading TensorRT engine from plan file..." << std::endl;
  std::ifstream planFile(planFilename); 

  if (!planFile.is_open())
  {
    std::cout << "Could not open plan file." << std::endl;
    return 1;
  }

  std::stringstream planBuffer;
  planBuffer << planFile.rdbuf();
  std::string plan = planBuffer.str();
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
  nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr);
  nvinfer1::IExecutionContext *context = engine->createExecutionContext();
  
  /* get the input / output dimensions */
  int inputBindingIndex, outputBindingIndex;
  inputBindingIndex = engine->getBindingIndex(inputName.c_str());
  outputBindingIndex = engine->getBindingIndex(outputName.c_str());

  if (inputBindingIndex < 0)
  {
    std::cout << "Invalid input name." << std::endl;
    return 1;
  }

  if (outputBindingIndex < 0)
  {
    std::cout << "Invalid output name." << std::endl;
    return 1;
  }

  nvinfer1::Dims inputDims, outputDims;
  inputDims = engine->getBindingDimensions(inputBindingIndex);
  outputDims = engine->getBindingDimensions(outputBindingIndex);

  std::cerr << "input b: " << inputDims.d[0] << std::endl;
  std::cerr << "input c: " << inputDims.d[1] << std::endl;
  std::cerr << "input h: " << inputDims.d[2] << std::endl;
  std::cerr << "input w: " << inputDims.d[3] << std::endl;

  int inputWidth, inputHeight;
  inputHeight = inputDims.d[2];
  inputWidth = inputDims.d[3];

  /* read image, convert color, and resize */
  std::cout << "read input image" << std::endl;
  cv::Mat image = cv::imread(imageFilename, CV_LOAD_IMAGE_COLOR);
  if (image.data == NULL)
  {
    std::cout << "Could not read image from file." << std::endl;
    return 1;
  }
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
  cv::resize(image, image, cv::Size(inputWidth, inputHeight));

  /* convert from uint8+NHWC to float+NCHW */
  float *inputDataHost, *outputDataHost;
  size_t numInput, numOutput;
  numInput = numTensorElements(inputDims);
  numOutput = numTensorElements(outputDims);
  inputDataHost = (float*) malloc(numInput * sizeof(float));
  outputDataHost = (float*) malloc(numOutput * sizeof(float));

  cvImageToTensor(image, inputDataHost, inputDims);
  std::vector<float> mean{0.485, 0.456, 0.406};
  std::vector<float> std{0.229, 0.224, 0.225};
  preprocessTensor(inputDataHost, inputDims, mean, std);

  /* transfer to device */
  float *inputDataDevice, *outputDataDevice;
  cudaMalloc((void**)&inputDataDevice, numInput * sizeof(float));
  cudaMalloc((void**)&outputDataDevice, numOutput * sizeof(float));
  cudaMemcpy(inputDataDevice, inputDataHost, numInput * sizeof(float), cudaMemcpyHostToDevice);
  void *bindings[2];
  bindings[inputBindingIndex] = (void*) inputDataDevice;
  bindings[outputBindingIndex] = (void*) outputDataDevice;

  /* execute engine */
  std::cout << "Executing inference engine..." << std::endl;
  const int kBatchSize = 1;
  context->execute(kBatchSize, bindings);

  /* transfer output back to host */
  cudaMemcpy(outputDataHost, outputDataDevice, numOutput * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "numOutput: " << numOutput << std::endl;

  float exp_sum = 0.0;
  for (int i=0; i<numOutput; ++i) {
    exp_sum += exp(outputDataHost[i]);
  }

  std::vector<float> probs;
  for (int i=0; i<numOutput; ++i) {
    float v = outputDataHost[i];
    v = exp(v) / exp_sum;
    probs.push_back(v);
    std::cout << probs[i] << std::endl;
  }

  /* parse output */
  std::vector<size_t> sortedIndices = argsort(outputDataHost, outputDims);

  std::cout << "\nThe top-5 indices are: ";
  for (int i = 0; i < 5; i++)
    std::cout << sortedIndices[i] << " ";

  std::ifstream labelsFile(labelFilename);
  if (!labelsFile.is_open())
  {
    std::cout << "\nCould not open label file." << std::endl;
    return 1;
  }
  std::vector<std::string> labelMap;
  std::string label;
  while(getline(labelsFile, label))
  {
    labelMap.push_back(label);
  }

  std::cout << "\nWhich corresponds to class labels: ";
  for (int i = 0; i < 5; i++)
    std::cout << std::endl << i << ": " << labelMap[sortedIndices[i]]
         << ", score: " << probs[sortedIndices[i]] * 100;
  std::cout << std::endl;

  /* clean up */
  std::cout << 0 << std::endl;
  runtime->destroy();

  std::cout << 0 << std::endl;
  engine->destroy();

  std::cout << 1 << std::endl;
  context->destroy();

  std::cout << 2 << std::endl;
  free(inputDataHost);

  std::cout << 3 << std::endl;
  free(outputDataHost);

  std::cout << 4 << std::endl;
  cudaFree(inputDataDevice);

  std::cout << 5 << std::endl;
  cudaFree(outputDataDevice);

  std::cout << "end proccess " << std::endl;

  return 0;
}
