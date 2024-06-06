# FacePrivacy

FacePrivacy is a script to blurr and anonymize facial features during a real time live stream instead of "Fix it in post".

This is a project for learning about [Yolov8](https://docs.ultralytics.com/modes/) and Roboflow's [Supervision](https://supervision.roboflow.com/latest/) library.




## Appendix

I used Ultralytics Yolov8 model for facial recognition due to its fast inference and accuerate results in Real time application even with limited hardware resources.

Collecting, organizing and preprocessing the dataset for training was done using [Roboflow](https://roboflow.com), and for handling the infernece ,blurring and showing Annotations i used Roboflow's Supervision library.

I also added a way to blurr and unblurr on the fly using Left mouse button




## Demo
https://github.com/MarwanM404/FacePrivacy/assets/102218690/2db8d6ad-3869-4a73-ab77-fd2785f9ba99

### TODO
1- Improve the dataset for detecting faces further in the background or out of focus

~~2- Testing performance using threading~~
