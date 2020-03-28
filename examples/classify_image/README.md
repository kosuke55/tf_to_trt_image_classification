# Classify Single Image 

# tensorrt inference sample
```
mkdir build
cd build
cmake ..
make
```

## build engine from onnx
```
./build/engine_builder_sample <input_onnx_file_path> <output_engine_file_path>
```

## inference
```
./build/classify_image_sample sample_images/blue.jpg ckpt.engine ckpt_label.txt input_0 output_0
```

## inference speed

### mobilenet

average value of 10 times using TensorRT.

| only execute | with data transfer | with image process |
| :--- | :---: | ---: |
| 0.814600 [ms] | 0.967000 [ms] | 3.202200 [ms] |

average value of 10 times using pytorch.

| only execute | with data transfer | with image process |
| :--- | :---: | ---: |
| 7.475567 [ms] |  | |
