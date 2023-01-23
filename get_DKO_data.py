'''
Script for extracting relevant info/data from behavior task raw data files
- OF = open field
- LD = light-dark
- SG = self-grooming assay

Data organization:
- d = mouse dictionary; keys are mouse IDs (e.g. '44AM_Rn'), values are nested dictionaries for tasks listed below
    - OFdf  (dataframe from ezTrack analysis of open field videos)
    - LDdf  (dataframe from idTracker anlaysis of light-dark videos)
    - SGdf  (dataframe from self-grooming duration scoring)
- SUMdf = summary dataframe; single-value measures from above tasks
    - Subj
    - OF % center
    - OF distance
    - LD % light
    - LD distance
    - LD transitions
    - SG duration
    - SG bouts
'''


import numpy as np
import pandas as pd
import os
#from scipy import stats
#from scipy import signal
from smallestenclosingcircle import * #for enclosing circle calculation
from math import sqrt, dist
from datetime import datetime, timedelta

# paths to raw data
OF_path = 'DKO_behavior/open_field/ezTrack_trajectories'  #contains csv's of ezTrack trajectories
SGpath = 'DKO_behavior/grooming/groom_times' #contains csv's of grooming scores
LDpath = 'DKO_behavior/light_dark/videos' #contains subfolders named after subjects; each subfolder contains trajectories.txt
LDpath2 = 'DKO_behavior/light_dark/NeverLeftDarkSide' # videos for mice that never left dark chamber so I didn't use idTrack

# initialize mouse dictionary
d = {}


########### Open Field ###########
print('loading open field data...')
# load OF ezTrack files (csv's)
for filename in os.listdir(OF_path):
    if filename.endswith('.csv'):
        f = filename.split(".")[0]
        name = f[:-15]  #split before _LocationOutput
        df = pd.read_csv(os.path.join(OF_path, filename))
        d[name] = {'OFdf' : df}

# combine all trajectory data
X=[]; Y=[]
for k in d.keys():
    location = d[k]['OFdf']
    x = list(location['X'])
    y = list(location['Y'])
    X+=x; Y+=y
points = [(x,y) for x,y in zip(X,Y)]

#find smallest circle that encloses all points; returns center coordinates (cx,cy) and radius (r)
cx, cy, r = make_circle(points)

#inner circle that comprises one third the arena (for half, change to sqrt(2)
center_r = r / sqrt(3)

#calculation proportion of time spent in center region
center_props = []
distance_m = []

for k in d.keys():
    location = d[k]['OFdf']
    inside_count=0
    for x,y in zip(location['X'], location['Y']):
        if (x - cx)**2 + (y - cy)**2 <= center_r**2: #if point is inside inner circle (or on circumfrence)
            inside_count += 1
    center_props.append(inside_count/14400 *100) #divide by total frames to get proportion of time spend in center
    
    distance = sum(location['Distance_px'])
    distance_m.append(distance * 0.42/217) #px-to-m conversion (tub diameter is ~42cm and ~217 pixels)
    
#create summary dataframe
SUMdf = pd.DataFrame(list(zip([k for k in d.keys()], center_props, distance_m)), columns =['Subj', 'OF % center', 'OF distance'])
print(len(d), 'subjects found')

########### Light-Dark ###########
'''
LD videos of animals that never entered the light chamber aren't in light_dark/videos, they are in light_dark/NeverLeftDarkSide; so % light, distance, and transitions will stay 0 (how those columns are initialized)

Dimensions of light chamber
- in pixels: 218-224 * 427-444 (measured using ezTrack notebook)
- in inches: 6 * ~12
'''
print('loading light-dark data...')
px_perim = 218+224+427+444
in_perim = 2*(0.3048+0.1524) #inches to m
conversion = in_perim/px_perim

total_frames = 29 * 600  #29fps * 10min; 29 is average framerate for WT test videos
SUMdf[['LD % light', 'LD distance', 'LD transitions']] = 0 #initialize new summary df columns

for dirs,subdirs,files in os.walk(LDpath, topdown=True):
    for name in subdirs:
        if not ((name=='segm')|(name=='.ipynb_checkpoints')): #get mouse folder names, ignore idTracker segm folder
            trajectories = pd.read_table(os.path.join(LDpath,name,'trajectories.txt'), delimiter='\t', usecols=(0,1), nrows=total_frames)
            d[name]['LDdf'] = trajectories #store in mouse dictionary
            
            #non-NaN values in trajectories df means animal detected i.e. in light side;
            #length is num frames with animal detected; divide by total frames to get proportion in light side
            percent_light = len(trajectories.dropna()) / total_frames
            SUMdf.loc[SUMdf['Subj']==name, 'LD % light'] = percent_light * 100
            
            SUMdf.loc[SUMdf['Subj']==name, 'LD distance'] = dist(trajectories['X1'].dropna(), trajectories['Y1'].dropna()) * conversion #get distance traveled in m
            
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
            
#for mice that didn't leave the dark chamber
for filename in os.listdir(LDpath2):
    name = filename.split(".")[0]
    d[name]['LDdf'] = None
    
    
########### Self-Grooming ###########
print('loading grooming data...')
SUMdf[['SG duration', 'SG bouts']] = 0 #initialize new summary df columns

for filename in os.listdir(SGpath):
    if filename.endswith('.csv'):
        name = filename.split(".")[0]
        df = pd.read_csv(os.path.join(SGpath, filename), header=None, usecols=(1,2), names=['duration','timestamp'])
        
        for i in range(len(df)): #convert timestamps to seconds
            D = datetime.strptime(df.duration[i], '%H:%M:%S.%f')
            deltaD = timedelta(minutes=D.minute, seconds=D.second, microseconds=D.microsecond)
            df.duration[i] = deltaD.total_seconds()
            T = datetime.strptime(df.timestamp[i], '%H:%M:%S.%f')
            deltaT = timedelta(minutes=T.minute, seconds=T.second, microseconds=T.microsecond)
            df.timestamp[i] = deltaT.total_seconds()
        
        d[name]['SGdf'] = df
        SUMdf.loc[SUMdf['Subj']==name, 'SG duration'] = sum(df.duration[1::2])  #1::2 because odd indices are grooming intervals, even are non-grooming intervals
        SUMdf.loc[SUMdf['Subj']==name, 'SG bouts'] = len(df.duration[1::2])
        
print('done C:')