## Compressed Results of [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
|Model|Type|Accuracy (%)|FLOPs (M)|Params (M)|Latency (ms)|Model Size (MB)|
|:---:|:---:|:---|:---|:---|:---|:---|
|VGG16|Original|74.00|629.20|15.30|71.65|59.91|
|VGG16|Compressed-1|71.89 (-2.11)|433.00 (1.45x)|5.16 (2.97x)|24.52 (2.91x)|20.19 (2.97x)|
|VGG16|Compressed-2|67.67 (-6.23)|212.74 (2.96x)|1.25 (12.19x)|11.34 (6.32x)|4.90 (12.23x)|
|MobileNetV2|Original|74.29|179.46|2.33|46.26|9.36|
|MobileNetV2|Compressed|73.42 (-0.87)|111.96 (1.60x)|0.79 (2.96x)|24.50 (1.89x)|3.33 (2.81x)|
|RepVGG|Original|76.44|1709.31|12.94|248.10|51.09|
|RepVGG|Compressed-1|74.80 (-1.64)|1637.71 (1.04x)|10.62 (1.22x)|113.35 (2.19x)|41.74 (1.22x)|
|RepVGG|Compressed-2|70.19 (-6.25)|725.71 (2.36x)|3.00 (4.32x)|51.69 (4.80x)|11.62 (4.40x)|


### Pre-trained models(onnx)

- [cifar100-mobilenetv2.onnx](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/torch/cifar100/pretrained/cifar100_mobilenetv2_x1_0.onnx)
- [cifar100-repvgg-a1.onnx](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/torch/cifar100/pretrained/cifar100_repvgg_a1.onnx)
- [cifar100-resnet56.onnx](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/torch/cifar100/pretrained/cifar100_resnet56.onnx)
- [cifar100-vgg16-bn.onnx](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/torch/cifar100/pretrained/cifar100_vgg16_bn.onnx)

### Compressed Models
- [VGG16_compressed1.pt](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/torch/cifar100/compressed/VGG16_compressed1.pt)
- [VGG16_compressed2.pt](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/torch/cifar100/compressed/VGG16_compressed2.pt)
- [MobileNetV2_compressed.pt](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/torch/cifar100/compressed/MobileNetV2_compressed.pt)
- [RepVGG_compressed1.pt](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/torch/cifar100/compressed/RepVGG_compressed1.pt)
- [RepVGG_compressed2.pt](https://netspresso-compression-toolkit-public.s3.ap-northeast-2.amazonaws.com/model_zoo/torch/cifar100/compressed/RepVGG_compressed2.pt)
