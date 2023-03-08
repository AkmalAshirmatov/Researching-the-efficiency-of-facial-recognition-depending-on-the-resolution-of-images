# Researching-the-efficiency-of-facial-recognition-depending-on-the-resolution-of-images

These experiments were a part of a course work.

For experiments LFW dataset was used, that was divided into train and validation sets. 

In the beginning several steps were done: 
- detected face with pretrained RetinaFace model 
- rotation by a random angle in the range of [-30, 30] degrees (the angle is selected from a uniform distribution) 
- random horizontal flipping

During experiments:
- 93.73% accuracy without compression
- 92.00% accuracy when compressed by 2 times
- 90.33% accuracy when compressed by 4 times
- 84.66% accuracy when compressed by 8 times
