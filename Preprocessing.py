import pandas as pd
import os

AUDIO_DIR = "./EmoDb_berlin_database/audio"
METADATA_DIR = "./EmoDb_berlin_database/metadata"

# getting filenames from AUDIO_DIR
# [0] is necessary as os.walk return list and this list is saved in another list
list_of_filenames = [filenames for (_, _, filenames) in os.walk(AUDIO_DIR)][0]
# extracting letters from filenames which encode particular emotion
list_of_emotions = [emotion[5] for emotion in list_of_filenames]

# decoding emotions names
for index, emotion in enumerate(list_of_emotions):
    if emotion == 'W':
        list_of_emotions[index] = 'anger'
    if emotion == 'L':
        list_of_emotions[index] = 'boredom'
    if emotion == 'E':
        list_of_emotions[index] = 'disgust'
    if emotion == 'A':
        list_of_emotions[index] = 'anxiety'
    if emotion == 'F':
        list_of_emotions[index] = 'happiness'
    if emotion == 'T':
        list_of_emotions[index] = 'sadness'
    if emotion == 'N':
        list_of_emotions[index] = 'neutral'

# creating csv file which stores filenames and corresponding to them emotions
dict = {'filename': list_of_filenames, 'emotion': list_of_emotions}
df = pd.DataFrame(dict)
df.to_csv(os.path.join(METADATA_DIR, 'EmoDB.csv'), index=False)
