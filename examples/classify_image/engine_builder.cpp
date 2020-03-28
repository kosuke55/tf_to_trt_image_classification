#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

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

  if (argc != 3) {
    std::cout << "Usage: engine_builder <input_onnx_file_path> <output_engine_file_path>" << std::endl;
    return 0;
  }

  std::string input_onnx_file_path = argv[1];
  std::string output_engine_file_path = argv[2];
  std::cout << "input_onnx_file_path: " << input_onnx_file_path << std::endl;
  std::cout << "output_engine_file_path: " << output_engine_file_path << std::endl;

  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
  nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

  nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
  if ( !parser->parseFromFile(input_onnx_file_path.c_str(),
                              static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)) ) {
    std::cerr << "failed to parse file" << std::endl;
    return -1;
  }

  size_t max_batch_size = 1;
  builder->setMaxBatchSize(max_batch_size);
  builder->setMaxWorkspaceSize(16 << 20);

  std::string precision = "fp16";
  if ( precision == "fp16" ) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if ( precision == "int8" ) {
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
  }

  nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
  if (!engine) {
    std::cerr << "failed to build engine" << std::endl;
    return -1;
  }
  // assert(network->getNbInputs() == 1);
  nvinfer1::Dims input_dims = network->getInput(0)->getDimensions();
  // assert(input_dims.nbDims == 4);

  // assert(network->getNvOutputs() == 1);
  nvinfer1::Dims output_dims = network->getOutput(0)->getDimensions();
  // assert(output_dims.nbDims == 2);

  // // save engine
  nvinfer1::IHostMemory *data = engine->serialize();
  std::ofstream file;

  file.open(output_engine_file_path, std::ios::binary | std::ios::out);
  if (!file.is_open()) {
    std::cerr << "failed to output engine file" << std::endl;
  }
  file.write((const char*)data->data(), data->size());
  file.close();

  return 0;
}
