import os


def download_model(device: str) -> os.path:
    if device == 'cuda':
        print('Using GPU')
        return '../weight/best_segment.pt'
    else:
        print('Using CPU')
        return '../weights/openvino'