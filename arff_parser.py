import arff
import numpy as np


class ArffParser:
    def load_file(self, file_path: str) -> None:
        self.dataset = np.array(arff.load(open(file_path))["data"])
    
    def get_data(self) -> list[tuple]:
        return [tuple(map(float, row)) for row in self.dataset[:, 0:2]]
    
    def get_clusters(self) -> np.ndarray:
        return self.dataset[:, 2]