import rasterio as rs
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def getRNIR_bands(nome_pasta):
  array_paths = [f'./BigEarthNet-v1.0/{nome_pasta}/{nome_pasta}_B04.tif' ,f'./BigEarthNet-v1.0/{nome_pasta}/{nome_pasta}_B08.tif']
  arr_band = []
  for i in range(0, len(array_paths)):
    arquivo = rs.open(array_paths[i]).read()
    arr_band.append(arquivo[0])
  return arr_band

def convert_labels(data_original, vetor_converter):
    label_convertido = ''
    for label_original, indice in data_original['original_labels'].items():
        if label_original in vetor_converter['labels']:
            for indice_conversao, vetor_conversao  in enumerate(data_original['label_conversion']):
                if (indice_conversao == indice):
                    dic = data_original['BigEarthNet-19_labels']
                    value = {i for i in dic if dic[i]==indice_conversao}
                    label_convertido = next(iter(value), None)
    return label_convertido
    
def salvarImagem_NDVI(nome_pasta_origem, nome_pasta_dest):
    bandas = getRNIR_bands(nome_pasta_origem)
    red_min, red_max = np.min(bandas[0]), np.max(bandas[0])
    nir_min, nir_max = np.min(bandas[1]), np.max(bandas[1])

    red_normalized = (bandas[0] - red_min) / (red_max - red_min)
    nir_normalized = (bandas[1] - nir_min) / (nir_max - nir_min)

    # Calculate NDVI
    
    ndvi = (nir_normalized - red_normalized) / (nir_normalized + red_normalized)
    with open(f'./BigEarthNet-v1.0/{nome_pasta_origem}/{nome_pasta_origem}_labels_metadata.json', 'r') as f:
        data1 = json.load(f)
    with open('label_indices.json', 'r') as f:
        data2 = json.load(f)
    if (convert_labels(data2, data1) != ''):
        plt.imsave(f'{nome_pasta_dest}/{convert_labels(data2, data1)}/{nome_pasta_origem}.jpg',ndvi, cmap='RdYlGn')  # Usar o mapa de cores RdYlGn para NDVI

def criarPasta(path, nome_pasta):
    pasta_label = os.path.join(path, nome_pasta)
    os.makedirs(pasta_label, exist_ok=True)

def separar_pastas():
  # Carregando o JSON
  with open('label_indices.json', 'r') as f:
      data = json.load(f)

  # Diretório de destino para organizar as imagens
  diretorio_dataset = './dataset/'
  criarPasta(diretorio_dataset, 'train')
  criarPasta(diretorio_dataset, 'test')
  criarPasta(diretorio_dataset, 'val')

  # Criar pastas com base nas labels do BigEarthNet-19_labels
  for label, indice in data['BigEarthNet-19_labels'].items():
      criarPasta(diretorio_dataset+"/train", label)
      criarPasta(diretorio_dataset+"/test", label)
      criarPasta(diretorio_dataset+"/val", label)

separar_pastas()

def locar_imagens():
    from random import shuffle
    from rasterio.errors import RasterioIOError
    pastas = os.listdir('./BigEarthNet-v1.0')
    shuffle(pastas)
    quantidade_treino = 38000
    quantidade_teste = 8000
    quantidade_val = 4000    
    train = pastas[0:quantidade_treino]
    test = pastas[quantidade_treino:quantidade_treino + quantidade_teste]
    val = pastas[quantidade_treino + quantidade_teste:quantidade_treino + quantidade_teste+quantidade_val]
    for folder in train:
        try:
            salvarImagem_NDVI(folder, f'./dataset/train')
        except RasterioIOError:
            pass
    for folder_test in test:
        try:
            salvarImagem_NDVI(folder_test, f'./dataset/test')
        except RasterioIOError:
            pass
    for folder_val in val:
        try:
            salvarImagem_NDVI(folder_val, f'./dataset/val')
        except RasterioIOError:
            pass
locar_imagens()
