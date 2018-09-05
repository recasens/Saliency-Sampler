## Saliency-Sampler
This is a PyTorch implementation of the paper [Learning to Zoom: a Saliency-Based Sampling Layer for Neural Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Adria_Recasens_Learning_to_Zoom_ECCV_2018_paper.pdf) by Adrià Recasens, Petr Kellnhofer, Simon Stent, Wojciech Matusik and Antonio Torralba.
The paper presents a saliency-based distortion layer for convolutional neural networks that helps to improve the spatial sampling of input data for a given task.



##Requirements
The implementation has been tested wihth PyTorch 0.4.0 but it is likely to work on previous version of PyTorch as well. 

##Citation
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