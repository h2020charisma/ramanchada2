from typing import List


def write_csv(x, y, delimiter=',') -> List[str]:
    return [f'{x[i]}{delimiter}{y[i]}\n' for i in range(len(x))]
