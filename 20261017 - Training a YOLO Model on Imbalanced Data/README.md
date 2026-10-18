# Training a YOLO Model on Imbalanced Data
## ABSTRACT
Real-world datasets are rarely balanced. While object detection models perform exceptionally well on frequent objects, they often struggle with rare categories due to a phenomenon known as the long-tail distribution. This workshop provides an overview of the class imbalance problem in computer vision and explores the strategies used to address it. Participants will learn to experiment with data-level interventions, such as image augmentations, and model-level interventions, such as injecting custom loss functions. By the end of the session, attendees will directly evaluate how these data-level and algorithm-level tweaks impact real-world detection performance.

## The outline of the workshop is as follows:
1. Understanding the long-tail problem in object detection datasets
2. Training a baseline model
3. Examining class imbalance through data and loss perspectives
4. Injecting custom re-weighing and loss functions
5. Evaluation and metrics comparison (mAP, Precision, Recall)

## Prerequisite knowledge
Attendees should be knowledgeable about these topics:
- Basic understanding of machine learning concepts
- Familiarity with object detection terminologies (bounding boxes, mAP, IoU)
- Fundamental understanding of CNN architectures
- Experience working with PyTorch and Jupyter Notebooks

## Dependencies
Dependencies needed for the Jupyter notebook:
- Python >= 3.10
- Pytorch (CUDA Version Recommended)
- Torchvision
- Ultralytics YOLO framework

## ABOUT THE SPEAKER
Marxandrea Diane G. Cabato is an undergraduate student at De La Salle University (DLSU), pursuing a degree in Computer Science as part of the honors program BSMSCS. Currently, she is conducting research on long-tail object detection, exploring model-level interventions to improve performance on tail classes.