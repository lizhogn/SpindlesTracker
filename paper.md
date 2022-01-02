# SpindleTracker: A Fully Automatic Tool for Spindle Tracking and analysis 

## Abstract
In this paper, we present a fully automated spindle tracking and analysis tool: SpindleTracker, which is used to track the dynamic mechanism of time-series spindles image in electron microscopy and to analyze the relationship between spindle length and time during mitosis. SpindleTracker is based on the principle of "tracking-by-detection". For spindle detection, we designed a new model structure YOLOX-SP, which can predict both spindle bounding box and endpoint calorimetric images. Using these two can be obtained to separate each spindle individual from the original image individually. The extraction of the spindle skeleton can be solved by solving the constrained optimization problem for the two endpoints of the spindle. For spindle tracking, we use the SORT algorithm which has excellent accuracy and speed. To the best of our knowledge, SpindleTracker is the first model that can fully automate the detection and analysis of spindle bodies. In terms of accuracy, SpindleTracker has also achieved a level comparable to manual annotation.

## Introduction
The spindle is a spindle-shaped cell structure formed during mitosis or meiosis in eukaryotic cells with a wide middle and narrow ends, mainly composed of a large number of longitudinally arranged microtubules. During cell division, cells assemble a spindle that captures the kinetochores and pulls them towards opposite poles to segregate the chromosomes faithfully. Studying the mechanism of spindle elongation is essential to understand the generation of some genetic diseases.

Manual tracking of spindles faces the same limitations as other biological problem, such as neuron segmentation and synapse annotations. In the past decade, advances in molecular cell biology have triggered the development of highly sophisticated live-cell fluorescence microscopy systems, meanwhile brings vast amounts of electronic imaging volume. While 
these data can lead scientists to new discoveries and understanding in milecular cell biology, they can also make them encounter greater challenges. Manually analyzing these data is inefficient and impractical. Therefore, it is increasingly importanct to develop automated tracking and analysis methods to replace manual analysis.

Although the deep learning boom has covered many areas of biology in recent years, and it has also generated landmark tools like AlphaFold2, which has greatly contributed to the development of biology. However, in some relatively cold areas where deep learning is tentatively out of reach, biologists are still using manual or semi-automatic tools for boring and inefficient parameter measurements.

In this paper, we focus on spindle measurement analysis and propose SpindleTracker, an fully automated spindle tracking and analysis tool. SpindleTracker follows the principle of "Tracking-by-Detection". Firstly,  spindles in each image are localized, and in order to be able to obtain the skeleton of each spindle, we designed a spindle location and endpoint detection network YOLOX-SP based on YOLOX. YOLOX-SP can simultaneously generate the bounding box and endpoint coordinates of each spindle in the image, and connect the two endpoints of each spindle through a " minimum path" constrained optimization problem to obtain a continuous single-pixel skeleton of the spindle. The real length of the spindle can be obtained from the number of skeleton pixel points and the microscope scale. For tracking, we use the SORT algorithm, which is commonly used in the field of multi-object tracking, and can effectively match the same target from frame to frame.

Since there is no previous open source dataset for spindle tracking, we chose budding yeast cells (an important model for learning the spindle mechanism) as our subject for review. The evaluation results show that our method outperforms all conventional methods, with 95% accuracy in target detection, 99% target tracking metrics, and within 2% error in spindle length measurement in the test set. The model and method can also be easily extended to other cells and similar microtubule structures, and we hope that SpindleTracker will become a benchmark for spindle morphology analysis and microtubule morphology analysis.

Our 3 contributions are as follows.
* We propose a new model for both spindle localization and spindle endpoint detection, which has excellent performance with less annotation and training cost.
* We designed a "minimal path" constrained optimization equation to solve the spindle skeleton morphology, which can effectively extract a single-pixel, continuous spindle skeleton and more accurately measure the spindle length and other information.
* We open source a new spindle tracking dataset and a fully automated spindle tracking and analysis tool, SpindleTracker.


## Relate Works
### Multi-Object Tracking
#### Objection Detection
In tracking-by-detection framework, object detection is the basis of multi-object tracking. In converlutional neural network(CNN)  based deep learning network, object detection algorithms can be divided into two categories according to detection stages: one is the two-stage model represented by Faster R-CNN~\cite{ren2016faster}, another is the single-stage model represented by YOLO~\cite{redmon2016you}. In early, the speed of one-stage model is faster than that of two-stage model, but the accuracy of the two-stage model is better than that of the one-stage model. However, due to the faster iterations of the YOLO series, one-stage model is now outperforming the two-stage model in terms of both speed and accuracy. The YOLOX-SP in this paper is a modification based on the latest YOLO series model: YOLOX, which has state-of-the-art accuracy in one-stage model.

#### Data Association
Data association is another core component of multi-object tracking, which first computes the similarity between tracklets and detection boxes and then matches them according to the similarity. In similarity metrics, location, motion and apperarance are useful cues for assocition. SORT combines location and motion cues, some recent methods design networks to learn object motions and achieve more robust results in cases of large camera motion or low frame rate. DeepSORT adopts a stand-alone Re-ID model to extract appearance features from the detection boxes. After similarity computation, matching strategy assigns identities to the objects. This can be done by Hungarian Algorithm or greedy assignment. SORT matches the detection boxes to the trackets by once matching. DeepSORT  proposes a cascaded matching strategy which first matches the detection boxes to the most recent tracklets and then to the lost ones. Due to the spindle simple motion and similiar appearance, SORT is fully competent.


### Spindle Analysis
#### Single spindle analysis
According to the number of spindles in the time-lapse imaging, it can be divided into single and multiple spindle tracking. Single spindle tracking are relatively simple, and off-the-shelf toolkits are more complete. Larson and Bement~\cite{larson2017automated} create a MATLAB toolbox for tracking the location and rotation angle of polar bodies in the mitotic spindle of epithelial cells in Xenopus laevis embryos. And Decarreau et al.~\cite{decarreau2014rapid} use MATLAB to semi-automatically track the rotation angle of the mammalian spindle, but required a large number of parameters to be entered form. Ana Sofia et al.~\cite{uzsoy2020automated} developed a tool to automatically calculate the length of the yeast spindle by projecting image onto the  MT's main axis where it is located and thus calculating the length between two endpoints.

#### Multi-spindle analysis
In multiple MT tracking, much of the previous work on MT tracking was done using an ImageJ plugin called MTrackJ~\cite{meijering2012methods}, a semi-automatic tool that requires manual annotation of MT's endpoints. Deep learning networks based on instance segmentation such as Mask R-CNN~\cite{he2018mask} are able to segment MTs efficiently and fit well in the field of multiple MT tracking. However, training a network for instance segmentation requires a large amount of unbearable labeling. Samira et al.~\cite{masoudi2020instance} proposed a new method for instance-level MT tracking using a recurrent attention network that effectively set into avoiding tedious manual labeling. However, designing a synthetic algorithm to make the generated data features approximate the distribution of real experimental data is not a simple task, which requires a good understanding of the imaging data set and is also time consuming.


## Method
The overview of the proposed spindle tracking and analysising pipeline is shown in Fig.\ref{fig:overview}. The input image sequence goes through a pipeline with the following 4 stages: 
(1) Image Preprocessing: the high-resolution image of the microscope is converted to an 8-bit image and the contrast is improved. 
(2) Object Detection: Input the preprocessing image to the YOLOX-SP model, we can get the spindle bounding boxes and endpoints heapmap.
(3) Skeleton Extracting: Through endpoint heapmap and bounding box, each spindle endpoint pair can be separated from the image, and then a single-pixel, continuous spindle skeleton curve is obtrained by solving a "minimum path" constrained optimization problem.
(4) Data Association:  The targets between frames are linked using SORT algorithm.

### Image Preprocessing
Unlike natural images, the data type of spindle images under electron microscope is generally stored in 16 bits. In order to be able to visualize and label the images better, the time-lapse images need to be normalized first. The minimum value of the image $I_{min}$ is generally mapped to 0 and the maximum value $I_{max}$ is mapped to 1 as follows:
$$
I_{normal} = \frac{I - I_{min}}{I_{max} - I_{min}}
$$
However, in such a mapping approach leads to insignificant contrast, which is increasing the difficulty of spindle detection. In order to obtain images with strong contrast, we compress the dynamic range of the image and the normalization function as follows:

$$
\begin{equation}
I_{normal} =
\left\{
	 \begin{array}{l}
	 0, & I_{normal} < 0 \\
	 \frac{I - (I_{min}+I_{interval}*t_1)}{I_{interval}(t_2-t_1)}, & 0\le I_{normal} \le 1\\

	 1, & I_{normal} > 1
	 \end{array}
\right.
\end{equation}
$$
where  $I_{interval}=I_{max} - I_{min}$ and  $t_1, t_2$ denote the positions of the upper and lower bounds of the maximum gray interval of the image. In spindle images, $[t_1, t_2]=[0, 0.75]$ has a good contrast effect.

### Object Detection
![[YOLO-SP model.png]]

YOLO-SP network architecture as shown in Fig 3，根据功能，According to the features, we can divide the YOLO-SP network into 2 parts: Spindle Detection and Heapmap Regression。
For spindle detection, we have completely inherited the structure of YOLOX. The backbone network is the same as YOLOv5 which adopts an advanced CSPNet backbone and an additional PAN head. There are two decoupled heads after the backbone network, one for regression and the other for classfication. An addtional IoU-aware branch is added to the regression head to predict the IoU between the predicted box and the ground truth box. The regression head directly predicts four values in each location in the feature map, i.e., two offsets in terms of the left-top corner of the grid, and the height and width of the predicted box. The regression head is supervised by GIoU loss and classification and IoU heads are supervised by the binary cross entropy loss.

In heatmap regression, 我们对YOLOX的Backbone进行了拓展，在FPN部分，我们继续进行特征图的上采样与融合操作，最终得到一个与输入图像尺寸相等的特征图，我们用这张特征图进行spindle endpoint的heatmap regression，目标是真实端点形成的高斯模糊图像，损失函数选用的是最小均方误差MSE。


### Skeleton Extracting


### Data Association





## Experiment


## Result



## Conclusion
