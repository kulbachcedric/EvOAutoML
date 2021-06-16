import os
from enum import Enum
from pathlib import Path

from skmultiflow.data import FileStream, AGRAWALGenerator, STAGGERGenerator, HyperplaneGenerator, LEDGenerator, \
    RandomRBFGenerator, SEAGenerator, ConceptDriftStream
from skmultiflow.data.base_stream import Stream
import pandas as pd

class StreamId(Enum):
    agrawal_gen = 1
    stagger_gen = 2
    hyperplane_gen = 3
    led_gen = 4
    rbf_gen = 5
    sea_gen = 6
    covtype = 7
    elec = 8
    pokerhand = 9
    conceptdrift_sea = 10
    conceptdrift_agrawal = 11
    conceptdrift_stagger = 12

DATA_PATH = './datasets/'
RANDOM_STATE_1 = 42
RANDOM_STATE_2 = 67
DATA_SIZE = 100000

def get_stream(streamId: str, from_cache:bool)-> Stream:
    if from_cache:
        path = Path(DATA_PATH + streamId +'.csv')
        if path.exists():
            return FileStream(filepath=str(path))
        else:
            generate_new_stream(streamId=streamId, n=DATA_SIZE)
            return FileStream(filepath=str(path))

def generate_new_stream(streamId:str, n=DATA_SIZE):
    generator = None
    if streamId == StreamId.agrawal_gen.name:
        generator =  AGRAWALGenerator()
    elif streamId == StreamId.stagger_gen.name:
        generator =  STAGGERGenerator(random_state=RANDOM_STATE_1)
    elif streamId == StreamId.hyperplane_gen.name:
        generator =  HyperplaneGenerator(random_state=RANDOM_STATE_1)
    elif streamId == StreamId.led_gen.name:
        generator =  LEDGenerator(random_state=RANDOM_STATE_1, has_noise=True, noise_percentage=.3)
    elif streamId == StreamId.rbf_gen.name:
        generator =  RandomRBFGenerator(model_random_state=RANDOM_STATE_1, sample_random_state=RANDOM_STATE_2, n_classes=10)
    elif streamId == StreamId.sea_gen.name:
        generator =  SEAGenerator(random_state=RANDOM_STATE_1)
    elif streamId == StreamId.conceptdrift_sea.name:
        generator = ConceptDriftStream(stream=SEAGenerator(classification_function=2, random_state=RANDOM_STATE_1), drift_stream=SEAGenerator(classification_function=3, random_state=RANDOM_STATE_2), position=10000)
    elif streamId == StreamId.conceptdrift_agrawal.name:
        generator = ConceptDriftStream(stream=AGRAWALGenerator(classification_function=0, random_state=RANDOM_STATE_1), drift_stream=AGRAWALGenerator(classification_function=4, random_state=RANDOM_STATE_2),position=10000)
    elif streamId == StreamId.conceptdrift_stagger.name:
        generator = ConceptDriftStream(STAGGERGenerator(classification_function=0, random_state=RANDOM_STATE_1), drift_stream=STAGGERGenerator(classification_function=2, random_state=RANDOM_STATE_2),position=10000)
    else:
        raise NotImplementedError('streamID: ' + streamId + 'is not implemented')
    X, y = generator.next_sample(batch_size=n)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y, dtype='int32')
    df = pd.concat([X, y], axis=1)
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    df.to_csv(f'{DATA_PATH}/{streamId}.csv', header=None, index=None,mode='w+')






