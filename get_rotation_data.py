'''
Script for extracting relevant info/data from behavior task raw data files; blinded to genotype
- SG = self-grooming
- MB = marble burying
- LD = light-dark

Data organization:
- d = mouse dictionary; keys are mouse IDs (e.g. '44AM_Rn'), values are nested dictionaries for tasks listed below
    - OFdf  (dataframe from ezTrack analysis of open field videos)
    - LDdf  (dataframe from idTracker anlaysis of light-dark videos)
    - SGdf  (dataframe from self-grooming duration scoring)
- SUMdf = summary dataframe; single-value measures from above tasks
    - Subj
    - SG duration
    - SG bouts
    - LD % light
    - LD distance
    - LD transitions
    
Usage: from get_rotation_data import d, SUMdf
'''

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  #turn off chained assignment warning
from matplotlib import pyplot as plt
from matplotlib import cm
from datetime import datetime, timedelta
import os
import scipy.stats as stats
import scipy.io
import seaborn as sns
from collections import defaultdict
from math import dist


# paths to raw data
groom_path = 'Rotation/Grooming/groom_times' #contains csv's of grooming scores
S3path = 'Rotation/Colony/actual_database/Syt3.csv'
S7path = 'Rotation/Colony/actual_database/Syt7.csv'
cage_ID_sheet = 'Rotation/Colony/cage_IDs.csv'
marb_path = 'Rotation/Marbles/Marbles_buried.xlsx'
LD_path = 'Rotation/Light-dark/analyzed' #contains subfolders named after subjects; each subfolder contains trajectories.txt

# initialize mouse dictionary
d = {}


########### Self-Grooming ###########
print('loading grooming data...')
# load grooming files
SG_dur = []
SG_bouts = []

for filename in os.listdir(groom_path):
    if filename.endswith('.csv'):
        f = filename.split(".")[0]
        name = f[10:]  #split after YYMMDD_CL
        df = pd.read_csv(os.path.join(groom_path, filename), header=None, usecols=(1,2), names=['duration','timestamp'])
        
        for i in range(len(df)): #convert timestamps to seconds
            D = datetime.strptime(df.duration[i], '%H:%M:%S.%f')
            deltaD = timedelta(minutes=D.minute, seconds=D.second, microseconds=D.microsecond)
            df.duration[i] = deltaD.total_seconds()
            T = datetime.strptime(df.timestamp[i], '%H:%M:%S.%f')
            deltaT = timedelta(minutes=T.minute, seconds=T.second, microseconds=T.microsecond)
            df.timestamp[i] = deltaT.total_seconds()

        d[name] = {'SGdf' : df}
        SG_dur.append(sum(df.duration[1::2])) #1::2 because odd indices are grooming intervals, even are non-grooming intervals
        SG_bouts.append(len(df.duration[1::2]))
        
#create summary dataframe
SUMdf = pd.DataFrame(list(zip([k for k in d.keys()], SG_dur, SG_bouts)), columns =['Subj', 'SG duration', 'SG bouts'])


########### Genotype data import ###########
print('fetching genotypes...')
#import mouse colony data
S3col = pd.read_csv(S3path, header=1, usecols=(2,4,5,6,7,8,9))
S7col = pd.read_csv(S7path, header=1, usecols=(2,4,5,6,7,8,9))
colonydata = pd.concat([S3col, S7col], ignore_index=True).dropna()

#import blinded cage IDs
cage_ID = pd.read_csv(cage_ID_sheet,index_col=None)

#extract experimental subject data from colony data
cagetags = [x[5:].replace('_','') for x in cage_ID.Cage.tolist()]
allcages = '|'.join(cagetags)
subjectdata = colonydata.loc[colonydata['Cage Tag'].str.contains(allcages)]
subjectdata['Sex'] = [x[0] for x in subjectdata['Sex']] #remove number from sex label

#add blinded cage IDs (Cage1, Cage2, etc) to experimental subject dataframe
for i in range(len(cage_ID)):
    [line, cage, sex] = cage_ID.Cage[i].split('_')
    subjectdata.loc[(subjectdata['Mouseline']==line)&(subjectdata['Cage Tag']==cage+sex), 'Cage ID'] = cage_ID.ID[i]
    
print(len(subjectdata), 'subjects found')
if len(d) > len(subjectdata):
    print('(subject data missing...)')

#add genotypes to SUMdf
SUMdf['Genotype'] = 0 #initialize new summary df column
for k in d.keys():
    [cage, notch] = k.split('_')
    row = subjectdata.loc[(subjectdata['Cage ID']==cage)&(subjectdata['Ear notch']==notch)]
    SUMdf.loc[SUMdf['Subj']==k, 'Genotype'] = row['Mouseline'].values[0] + row['Genotype'].values[0] #gives string, e.g. 'Syt3-/-'


########### Marble Burying ###########
print('loading marble data...')
#import marble data
marbles = pd.read_excel(marb_path, usecols=(0,3))
SUMdf['Marbles'] = 0 #initialize new summary df column

#add number of buried marbles to summary dataframe
for i in range(len(marbles)):
    name = marbles['Subject'][i]
    SUMdf.loc[SUMdf['Subj']==name, 'Marbles'] = marbles['Marbles buried'][i]
    

########### Light-Dark ###########
'''
idTracker puts all results (segm folder, trajectories.mat, trajectories.txt) into the folder containing the video;
organize batch file moves all files into their own folder with the same filename;
put new vids in idTracker folder and batch them into folders; move to analyzed folder once analyzed

Dimensions of light chamber
- in pixels: 276-288 * 551-557 (measured using ezTrack notebook)
- in inches: 6 * ~12
'''
print('loading light-dark data...')
px_perim = 276+288+551+557
in_perim = 2*(0.3048+0.1524) #inches to m
conversion = in_perim/px_perim

# OLD
total_frames = 29 * 600  #29fps * 10min; 29 is average framerate for WT test videos
SUMdf[['LD % light', 'LD distance', 'LD transitions']] = 0 #initialize new summary df columns

for dirs,subdirs,files in os.walk(LD_path, topdown=True):
    for subfol in subdirs:
        if not ((subfol=='segm')|(subfol=='.ipynb_checkpoints')): #get mouse folder names, ignore other folders
            trajectories = pd.read_table(os.path.join(LD_path,subfol,'trajectories.txt'), delimiter='\t', usecols=(0,1), nrows=total_frames)
            name = subfol[7:]  #split after YYMMDD_
            d[name]['LDdf'] = trajectories #store in mouse dictionary
            
            #non-NaN values in trajectories df means animal detected i.e. in light side;
            #length is num frames with animal detected; divide by total frames to get proportion in light side
            percent_light = len(trajectories.dropna()) / total_frames
            SUMdf.loc[SUMdf['Subj']==name, 'LD % light'] = percent_light * 100
            
            #get distance traveled in m
            SUMdf.loc[SUMdf['Subj']==name, 'LD distance'] = dist(trajectories['X1'].dropna(), trajectories['Y1'].dropna()) * conversion
            
            #get number of transitions between chambers
            trans_count = 0
            previous = trajectories['X1'][0]
            for i,f in enumerate(trajectories['X1']):
                if i==0: continue  #skip first frame
                if np.isnan(trajectories['X1'][i]): current = 1 #if NaN, make int
                else: current = 'c'  #if not NaN, make string
                if type(current) is not type(previous):
                    trans_count += 1 #if type changed, transition occurred
                previous = current
            SUMdf.loc[SUMdf['Subj']==name, 'LD transitions'] = trans_count

print('done :)')