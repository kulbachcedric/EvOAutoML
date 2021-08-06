from river import stream
from river.datasets import base


class Covtype(base.RemoteDataset):

    def __init__(self):
        super().__init__(
            url="https://datahub.io/machine-learning/covertype/r/covertype.csv",
            size=103930879,
            unpack=False,
            task=base.MULTI_CLF,
            n_samples=58_1012,
            n_features=54,
            filename="covertype.csv",
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target="class",
            converters={
                'Elevation': float,
                'Aspect': float,
                'Slope': float,
                'Horizontal_Distance_To_Hydrology': float,
                'Vertical_Distance_To_Hydrology': float,
                'Horizontal_Distance_To_Roadways': float,
                'Hillshade_9am': float,
                'Hillshade_Noon': float,
                'Hillshade_3pm': float,
                'Horizontal_Distance_To_Fire_Points': float,
                'Wilderness_Area1': int,
                'Wilderness_Area2': int,
                'Wilderness_Area3': int,
                'Wilderness_Area4': int,
                'Soil_Type1': int,
                'Soil_Type2': int,
                'Soil_Type3': int,
                'Soil_Type4': int,
                'Soil_Type5': int,
                'Soil_Type6': int,
                'Soil_Type7': int,
                'Soil_Type8': int,
                'Soil_Type9': int,
                'Soil_Type10': int,
                'Soil_Type11': int,
                'Soil_Type12': int,
                'Soil_Type13': int,
                'Soil_Type14': int,
                'Soil_Type15': int,
                'Soil_Type16': int,
                'Soil_Type17': int,
                'Soil_Type18': int,
                'Soil_Type19': int,
                'Soil_Type20': int,
                'Soil_Type21': int,
                'Soil_Type22': int,
                'Soil_Type23': int,
                'Soil_Type24': int,
                'Soil_Type25': int,
                'Soil_Type26': int,
                'Soil_Type27': int,
                'Soil_Type28': int,
                'Soil_Type29': int,
                'Soil_Type30': int,
                'Soil_Type31': int,
                'Soil_Type32': int,
                'Soil_Type33': int,
                'Soil_Type34': int,
                'Soil_Type35': int,
                'Soil_Type36': int,
                'Soil_Type37': int,
                'Soil_Type38': int,
                'Soil_Type39': int,
                'Soil_Type40' :int,
        }
        )

class PokerHand(base.RemoteDataset):
    def __init__(self):
        super().__init__(
            url="http://nrvis.com/data/mldata/poker-hand.csv",
            size=588_957,
            unpack=False,
            task= base.MULTI_CLF,
            n_samples=1_025_010,
            n_features=11,
            filename='poker-hand.csv'
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target="class",
            converters={
            }
        )