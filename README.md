## Saliency-Sampler
This is the official PyTorch implementation of the paper [Learning to Zoom: a Saliency-Based Sampling Layer for Neural Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Adria_Recasens_Learning_to_Zoom_ECCV_2018_paper.pdf) by: 

<div align="center">
<img src="https://raw.githubusercontent.com/recasens/Saliency-Sampler/master/images/author_list.png" height="160px">
</div>


The paper presents a saliency-based distortion layer for convolutional neural networks that helps to improve the spatial sampling of input data for a given task. For instance, for the gaze-tracking task and the fine-grained classification task, the layer produces deformed images such as:

<div align="center">
<img src="https://raw.githubusercontent.com/recasens/Saliency-Sampler/master/images/augmentation_image.png" height="160px">
</div>


## Requirements
The implementation has been tested with PyTorch 0.4.0 but it is likely to work on previous versions of PyTorch as well. 

## Usage
To add a Saliency Sampler layer at the beginning of your model, you just need to define a task network and a saliency network and instantiate the model as:
```
task_network = resnet101()
saliency_network = saliency_network_resnet18()
task_input_size = 224
saliency_input_size = 224
model = Saliency_Sampler(task_network,saliency_network,task_input_size,saliency_input_size)
```


## Citation
If you want to cite our research, please use:
```
@inproceedings{recasens2018learning,
  title={Learning to Zoom: a Saliency-Based Sampling Layer for Neural Networks},
  author={Recasens, Adria and Kellnhofer, Petr and Stent, Simon and Matusik, Wojciech and Torralba, Antonio},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={51--66},
  year={2018}
}
```
