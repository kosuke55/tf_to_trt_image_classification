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

int main(int argc, char *argv[])
{
  if (argc != 6) {
    std::cout << "Usage: classify_image <image_file> <plan_file> <label_file> <input_name> <output_name>";
    return 0;
  }

  std::string imageFilename = argv[1];
  std::string planFilename = argv[2];
  std::string labelFilename = argv[3];
  std::string inputName = argv[4];
  std::string outputName = argv[5];

  // read label
  std::vector<std::string> labels;
  read_labelfile(labelFilename, labels);

  /* load the engine */
  std::cout << "Loading TensorRT engine from plan file..." << std::endl;
  std::ifstream planFile(planFilename); 
  if (!planFile.is_open()) {
    std::cout << "Could not open plan file." << std::endl;
    return -1;
  }

  std::stringstream planBuffer;
  planBuffer << planFile.rdbuf();
  std::string plan = planBuffer.str();
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
  nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr);
  nvinfer1::IExecutionContext *context = engine->createExecutionContext();
  
  int inputBindingIndex = engine->getBindingIndex(inputName.c_str());
  int outputBindingIndex = engine->getBindingIndex(outputName.c_str());
  if (inputBindingIndex < 0) {
    std::cout << "Invalid input name." << std::endl;
    return -1;
  }
  if (outputBindingIndex < 0) {
    std::cout << "Invalid output name." << std::endl;
    return -1;
  }

  nvinfer1::Dims inputDims = engine->getBindingDimensions(inputBindingIndex);
  nvinfer1::Dims outputDims = engine->getBindingDimensions(outputBindingIndex);
  int inputWidth = inputDims.d[2];
  int inputHeight = inputDims.d[3];

  /* convert from uint8+NHWC to float+NCHW */
  size_t numInput = numTensorElements(inputDims);
  size_t numOutput = numTensorElements(outputDims);
  float *inputDataHost = (float*) malloc(numInput * sizeof(float));
  float *outputDataHost = (float*) malloc(numOutput * sizeof(float));

  /* read image, convert color, and resize */
  cv::Mat image = cv::imread(imageFilename, CV_LOAD_IMAGE_COLOR);
  if (image.data == NULL) {
    return -1;
  }

  int execute_num = 10;

  double total_elapsed_time_involve_execute;
  double total_elapsed_time_involve_data_transfer;
  double total_elapsed_time_involve_image_process;

  std::chrono::system_clock::time_point start_involve_image_process, end_involve_image_process;
  std::chrono::system_clock::time_point start_involve_data_transfer, end_involve_data_transfer;
  std::chrono::system_clock::time_point start_involve_execute, end_involve_execute;

  for (int i=0; i<execute_num; i++) {
    printf(" --- %d\n", i);

    start_involve_image_process = std::chrono::system_clock::now();

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
    cv::resize(image, image, cv::Size(inputWidth, inputHeight));
    cvImageToTensor(image, inputDataHost, inputDims);
    std::vector<float> mean{0.485, 0.456, 0.406};
    std::vector<float> std{0.229, 0.224, 0.225};
    preprocessTensor(inputDataHost, inputDims, mean, std);

    start_involve_data_transfer = std::chrono::system_clock::now();
    /* transfer to device */
    float *inputDataDevice;
    cudaMalloc((void**)&inputDataDevice,
               numInput * sizeof(float));
    cudaMemcpy(inputDataDevice,
               inputDataHost,
               numInput * sizeof(float),
               cudaMemcpyHostToDevice);
    float *outputDataDevice;
    cudaMalloc((void**)&outputDataDevice,
               numOutput * sizeof(float));
    void *bindings[2];
    bindings[inputBindingIndex] = (void*) inputDataDevice;
    bindings[outputBindingIndex] = (void*) outputDataDevice;

    /* execute engine */
    start_involve_execute = std::chrono::system_clock::now();
    context->executeV2(bindings);
    end_involve_execute = std::chrono::system_clock::now();

    /* transfer output back to host */
    cudaMemcpy(outputDataHost,
               outputDataDevice,
               numOutput * sizeof(float),
               cudaMemcpyDeviceToHost);
    std::vector<float> probs;
    calc_softmax(outputDataHost, numOutput, probs);
    std::vector<size_t> sortedIndices = argsort(outputDataHost, outputDims);
    // printf("label: %s, score: %.2f\%\n",
    //        labels[sortedIndices[0]].c_str(),
    //        probs[sortedIndices[0]] * 100);
    cudaFree(inputDataDevice);
    cudaFree(outputDataDevice);

    end_involve_data_transfer = std::chrono::system_clock::now();
    end_involve_image_process = std::chrono::system_clock::now();

    double elapsed_time_involve_execute = static_cast<double>
      (std::chrono::duration_cast<std::chrono::microseconds>
       (end_involve_execute - start_involve_execute).count()) / 1000;
    double elapsed_time_involve_data_transfer = static_cast<double>
      (std::chrono::duration_cast<std::chrono::microseconds>
       (end_involve_data_transfer - start_involve_data_transfer).count()) / 1000;
    double elapsed_time_involve_image_process = static_cast<double>
      (std::chrono::duration_cast<std::chrono::microseconds>
       (end_involve_image_process - start_involve_image_process).count()) / 1000;


    printf("elapsed_time_involve_execute: %lf [ms]\n", elapsed_time_involve_execute);
    printf("elapsed_time_involve_data_transfer: %lf [ms]\n", elapsed_time_involve_data_transfer);
    printf("elapsed_time_involve_image_process: %lf [ms]\n", elapsed_time_involve_image_process);

    total_elapsed_time_involve_execute += elapsed_time_involve_execute;
    total_elapsed_time_involve_data_transfer += elapsed_time_involve_data_transfer;
    total_elapsed_time_involve_image_process += elapsed_time_involve_image_process;

  }

  printf(" --- average\n");
  double average_elapsed_time_involve_execute = total_elapsed_time_involve_execute / execute_num;
  double average_elapsed_time_involve_data_transfer = total_elapsed_time_involve_data_transfer / execute_num;
  double average_elapsed_time_involve_image_process = total_elapsed_time_involve_image_process / execute_num;
  printf("average_elapsed_time_involve_execute: %lf [ms]\n", average_elapsed_time_involve_execute);
  printf("average_elapsed_time_involve_data_transfer: %lf [ms]\n", average_elapsed_time_involve_data_transfer);
  printf("average_elapsed_time_involve_image_process: %lf [ms]\n", average_elapsed_time_involve_image_process);


  runtime->destroy();
  engine->destroy();
  free(inputDataHost);
  free(outputDataHost);

  return 0;
}
