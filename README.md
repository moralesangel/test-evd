# emergency-vehicles-detection
This Deep Learning project is develop to detect emergency vehicles (ambulance, police cars, ...) using street audio. CNN and chroma features will help us achieve this goal.

## Getting data
We will use this dataset for our purposes: https://research.google.com/audioset/dataset/index.html
This dataset give us Youtube urls to vides about different noises, like helicopters, wind, car engines, traffic noise, ... and of course emergency vehicle noises.

We will create a Python program to dowload each emergency vehicle video, get just the clip we want (the dataset give us start and finish time of the clip), and get rid of the image part, since we are only going to work with audio.

Then, we will use those relevant classes in the dataset, like those mentioned before (we are not going to train this CNN with pig noises, but they are in the dataset anyways), to have 50% of emergency vehicle audios and 50% of anything else.

## Preprocessing
Some clips are 3 seconds long, an others 10 seconds long, so we need to adapt them to have the same duration. Since most of them are 10 seconds long, we will adapt them all to this duration. We will also normalize this audio values.

Now, as we are going to work with CNN, we are going to convert those audio files into images, by extrating their chroma features.
Chroma features are the relevance of each musical pitch (A to G or Do to Si depending on the country) in a specific moment of the audio. Let's see two examples:

![Idle Engine](images/engine.png)
![Girl singing](images/singing.png)

The first one is an idle engine, so same frequency over time. The second one is a girl singing, so different pitches.


## 
