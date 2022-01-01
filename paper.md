# SpindleTracker: A Fully Automatic Tool for Spindle Tracking and analysis 

## Abstract
In this paper, we present a fully automated spindle tracking and analysis tool: SpindleTracker, which is used to track the dynamic mechanism of time-series spindles image in electron microscopy and to analyze the relationship between spindle length and time during mitosis. SpindleTracker is based on the principle of "tracking-by-detection". For spindle detection, we designed a new model structure YOLOX-SP, which can predict both spindle bounding box and endpoint calorimetric images. Using these two can be obtained to separate each spindle individual from the original image individually. The extraction of the spindle skeleton can be solved by solving the constrained optimization problem for the two endpoints of the spindle. For spindle tracking, we use the SORT algorithm which has excellent accuracy and speed. To the best of our knowledge, SpindleTracker is the first model that can fully automate the detection and analysis of spindle bodies. In terms of accuracy, SpindleTracker has also achieved a level comparable to manual annotation.

## Introduction
The spindle is a spindle-shaped cell structure formed during mitosis or meiosis in eukaryotic cells with a wide middle and narrow ends, mainly composed of a large number of longitudinally arranged microtubules. During cell division, cells assemble a spindle that captures the kinetochores and pulls them towards opposite poles to segregate the chromosomes faithfully. Studying the mechanism of spindle elongation is essential to understand the generation of some genetic diseases.

Manual tracking of spindles faces the same limitations as other biological problem, such as neuron segmentation and synapse annotations. In the past decade, advances in molecular cell biology have triggered the development of highly sophisticated live-cell fluorescence microscopy systems, meanwhile brings vast amounts of electronic imaging volume. While 
these data can lead scientists to new discoveries and understanding in milecular cell biology, they can also make them encounter greater challenges. Manually analyzing these data is inefficient and impractical. Therefore, it is increasingly importanct to develop automated tracking and analysis methods to replace manual analysis.

Although the deep learning boom has covered many areas of biology in recent years, and it has also generated great tools like AlphaFold2, which has greatly contributed to the development of biology. However, in some relatively cold areas where deep learning is tentatively out of reach, biologists are still using manual or semi-automatic tools for boring and inefficient parameter measurements.

In this paper, we focus on automated spindle measurement analysis and propose SpindleTracker, an fully automated spindle tracking and analysis tool. SpindleTracker follows the principle of "Tracking-by-Detection". The spindle in each image is first localized, and in order to be able to obtain the skeleton of each spindle in the image, we designed the spindle position detection and endpoint detection network YOLOX-SP based on YOLOX. YOLOX-SP can simultaneously generate the bounding box and endpoint coordinates of each spindle in the image, and connect the two endpoints of each spindle through a " minimum path" constrained optimization problem to obtain a continuous single-pixel skeleton of the spindle. Information on the real length of the spindle can be obtained from the number of skeleton pixel points and the microscope scale. For tracking, we use the SORT algorithm, which is commonly used in the field of computer vision, and can effectively match the same target from frame to frame.

Since there is no previous open source dataset for spindle tracking, we chose budding yeast cells (an important model for learning the spindle mechanism) as our subject for review. The evaluation results show that our method outperforms all conventional methods, with 95% accuracy in target detection, 99% target tracking metrics, and within 2% error in spindle length measurement in the test set. The model and method can also be easily extended to other cells and similar microtubule structures, and we hope that SpindleTracker will become a benchmark for spindle morphology analysis and microtubule morphology analysis.

Our 3 contributions are as follows.
* We propose a new model for both spindle localization and spindle endpoint detection, which has excellent performance with less annotation and training cost.
* We designed a "minimal path" constrained optimization equation to solve the spindle skeleton morphology, which can effectively extract a single-pixel, continuous spindle skeleton and more accurately measure the spindle length and other information.
* We open source a new spindle tracking dataset and a fully automated spindle tracking and analysis tool, SpindleTracker.


## Relate Works
### Spindle Analysis


### Object Detection



## Method


## Result

