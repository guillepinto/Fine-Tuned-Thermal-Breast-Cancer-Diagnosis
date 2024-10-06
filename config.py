from pathlib import Path
import torch

# Load environment variables from .env file if it exists
PROJ_ROOT = Path(__file__).resolve().parent

print(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = PROJ_ROOT / "Imagens e Matrizes da Tese de Thiago Alves Elias da Silva"

TEST_PATH = DATA_DIR / "12 Novos Casos de Testes"
TRAIN_PATH =  DATA_DIR / "Desenvolvimento da Metodologia"
