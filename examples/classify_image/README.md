# Classify Single Image 

# mobilenetv2 sample
```
mkdir build
cd build
cmake ..
make
```

```
./build/classify_image_sample red.jpg ckpt.engine ckpt_label.txt input_0 output_0
```

## inference speed

### Tensorrt

average value of 10 times.

| only execute | with data transfer | with image process |
| :--- | :---: | ---: |
| 0.814600 [ms] | 0.967000 [ms] | 3.202200 [ms] |

### pytorch

average value of 10 times.

| only execute | with data transfer | with image process |
| :--- | :---: | ---: |
| 7.475567 [ms] |  | |

