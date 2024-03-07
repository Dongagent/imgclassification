import gdown

url = 'https://drive.google.com/file/d/11ozVs6zByFjs9viD3VIIP6qKFgjZwv9E/view?usp=sharing'
output_path = 'screws.zip'
gdown.download(url, output_path, quiet=False,fuzzy=True)


