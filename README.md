# Automatic-Car-Tracker
automatic car tracker using YOLO4 and siammask tracker algorithm

SiamMask is a deep learning model architecture that performs both Visual Object
Tracking (VOT) and semi-supervised Video Object Segmentation (VOS). Given
the location of the object in the first frame of the sequence, the aim of VOT is to
estimate an object's position in subsequent frames with the best possible
accuracy. Similarly, the main goal of VOS is to output a binary segmentation
6
mask that expresses whether or not a pixel belongs to the target. In other words,
SiamMask takes as input a single object bounding box for initialization and
outputs segmentation mask and object bounding box for each subsequent frame
of a video.
My implementation steps are as follows:
First, through the downloaded weights of the YOLO4 model, I detected all the
cars in the first frame.
Then, by use of the SiamMask algorithm, In the rest of the frames, the bounding
boxes are automatically added to the frame and finally produce the output video.
Although in this method, the target object must be YOLO-friendly, it is not difficult
to change it so that it becomes suitable for all sorts of annotating or
object-tracking problems 
