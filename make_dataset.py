import pandas as pd # 1.5.3
import os 
import numpy as np  # 1.24.4
from torch.utils.data import Dataset, Subset, DataLoader # 2.4.0
from torchvision.transforms import v2 # 0.19.0
from sklearn.model_selection import GroupKFold # 1.4.2
from PIL import Image, ImageOps # 10.3.0
from preprocess import crop_breast

TEST_PATH = "Imagens e Matrizes da Tese de Thiago Alves Elias da Silva/12 Novos Casos de Testes"
TRAIN_PATH = "Imagens e Matrizes da Tese de Thiago Alves Elias da Silva/Desenvolvimento da Metodologia"

def make_dataframe(train_path=TRAIN_PATH, test_path=TEST_PATH):

  patients = []
  labels = []
  segmented_images = []
  matrices = []

  """
  Esta construcción del dataset depende de la estructura del mismo
  """

  # Primero consigo la ruta de imagenes y matrices para cada uno de los pacientes

  for category in os.listdir(test_path):
    # print(category)
    for patient in os.listdir(os.path.join(test_path, category)):
      patient_path = os.path.join(test_path, category, patient)
      # print(patient_path)
      for record in os.listdir(f'{patient_path}/Segmentadas'):
        record_path = os.path.join(f'{patient_path}/Segmentadas', record)
        # print(record_path)
        segmented_images.append(record_path)
        if '-dir.png' in record_path:
          matrix_path = os.path.join(record_path.replace('Segmentadas','Matrizes').replace("-dir.png", ".txt"))
        elif '-esq.png' in record_path:
          matrix_path = os.path.join(record_path.replace('Segmentadas','Matrizes').replace("-esq.png", ".txt"))
        # print(matrix_path)
        if os.path.exists(matrix_path):
          matrices.append(matrix_path)
        else:
          good_part, bad_part = matrix_path[:len(matrix_path)//2], matrix_path[len(matrix_path)//2:]
          bad_part = bad_part.replace('Matrizes', 'Matrizes de Temperatura')
          matrix_path = good_part+bad_part
          matrices.append(matrix_path)
          # print(matrix_path)

        label = patient_path.split('/')[2]
        if label == 'DOENTES':
          label = 1
        else:
          label = 0
        labels.append(label)
        patients.append(record.split('_')[1])

  for category in os.listdir(train_path):
    # print(category)
    for patient in os.listdir(os.path.join(train_path, category)):
      patient_path = os.path.join(train_path, category, patient)
      for record in os.listdir(f'{patient_path}/Segmentadas'):
        record_path = os.path.join(f'{patient_path}/Segmentadas', record)
        # print(record_path)
        segmented_images.append(record_path)
        if '-dir.png' in record_path:
          matrix_path = os.path.join(record_path.replace('Segmentadas','Matrizes').replace("-dir.png", ".txt"))
        elif '-esq.png' in record_path:
          matrix_path = os.path.join(record_path.replace('Segmentadas','Matrizes').replace("-esq.png", ".txt"))
        # print(matrix_path)
        if os.path.exists(matrix_path):
          matrices.append(matrix_path)
        else:
          good_part, bad_part = matrix_path[:len(matrix_path)//2], matrix_path[len(matrix_path)//2:]
          bad_part = bad_part.replace('Matrizes', 'Matrizes de Temperatura')
          matrix_path = good_part+bad_part
          matrices.append(matrix_path)
          # print(matrix_path)

        label = patient_path.split('/')[2]
        if label == 'DOENTES':
          label = 1
        else:
          label = 0
        labels.append(label)
        patients.append(record.split('_')[1])

  # Crear un DataFrame con la información
  data = pd.DataFrame({
      'patient': patients,
      'segmented_image': segmented_images,
      'matrix': matrices,
      'label': labels
  })

  return data

def make_folds(data:pd.DataFrame):
    # Extraer los datos para GroupKFold
    X = np.array([i for i in range(len(data))])
    y = data['label'].values
    groups = data['patient'].values

    folds_dict = {}
    groupk_folds = 7
    gkf = GroupKFold(n_splits=groupk_folds)

    # Realizar la validación cruzada por grupos
    for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups), 1):        
        fold_name = f"fold_{i}"
        folds_dict[fold_name] = {
            'train': train_index,
            'test': test_index
        }

    return folds_dict

def make_subdataframes(data:pd.DataFrame, folds:dict):
  # Crear subdataframes
  subdataframes = {}

  for fold_name, indices in folds.items():
      train_df = data.iloc[indices['train']]
      test_df = data.iloc[indices['test']]
      
      subdataframes[fold_name] = {
          'train': train_df,
          'test': test_df
      }
  
  return subdataframes

MAX_TEMPERATURE = 36.425987

class ThermalDataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame, n_channels:int=1, transform=None, normalize: bool = None):
        self.dataframe = dataframe
        self.n_channels = n_channels
        self.normalize = normalize
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        """ Carga de la imagen """

        # Entramos a la carpeta y conseguimos la imagen de la lista
        img_path = self.dataframe.iloc[index]['segmented_image']

        # Leemos la imagen segmentada en escala de grises
        img = Image.open(img_path)
        img = ImageOps.grayscale(img)
        img = np.array(img)

        """ Carga de la matrix """

        matrix_path = self.dataframe.iloc[index]['matrix']
        matrix = np.loadtxt(matrix_path, dtype=np.float32)

        """ Consigo la imagen segmentada con los valores de la matrix """

        segmented = np.where(img == 0, 0, 1)
        img = (matrix * segmented).astype(np.float32) # float32, shape (480, 640)
        img = crop_breast(img) # cropped and resized to (224, 224)

        # Le agrego un canal explícito
        img = np.expand_dims(img, axis=2) # img tiene forma (224, 224, 1)

        # Replicar el canal si n_channels es 3
        if self.n_channels == 3:
            img = np.repeat(img, 3, axis=2) # img tiene forma (224, 224, 3)

        if self.normalize:
            img /= MAX_TEMPERATURE

        """ Consiguiendo el label """

        label = self.dataframe.iloc[index]['label']

        """ Convertir las imágenes en tensores, data augmentation y hacer resize """
        if self.transform:
            # Aplicamos las transformaciones a la imagen
            img = self.transform(img)

        return img, label

def get_data(transform, normalize:bool=False, slices:int=1, fold:int=None, n_channels:int=1):

    data = make_dataframe()

    # Generate folds
    folds = make_folds(data)

    # Create subdataframes  
    subdataframes = make_subdataframes(data, folds)

    if not fold:
      fold = np.random.choice(range(1, 8))

    fold_name = f'fold_{fold}'
    print(f"FOLD {fold}\n-------------------------------")

    train_dataset = ThermalDataset(subdataframes[fold_name]['train'],
                                    transform=transform, normalize=normalize,
                                    n_channels=n_channels)
    test_dataset = ThermalDataset(subdataframes[fold_name]['test'],
                                    transform=v2.ToImage(), normalize=normalize,
                                    n_channels=n_channels)
    
    # test with less data, it helped me to set up the experiments faster if slice=1
    # then it returns the complete dataset
    train_dataset = Subset(train_dataset, 
                                            indices=range(0, len(train_dataset), slices))
    test_dataset = Subset(test_dataset, 
                                            indices=range(0, len(test_dataset), slices))

    # return train_dataset, val_dataset, test_dataset
    return train_dataset, test_dataset

def make_loader(dataset, batch_size):
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True, num_workers=2)
    return loader