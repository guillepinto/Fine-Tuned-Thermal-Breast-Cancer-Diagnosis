from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
PROJ_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJ_ROOT/'.env')

print(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "Imagens e Matrizes da Tese de Thiago Alves Elias da Silva"

TEST_PATH = DATA_DIR / "12 Novos Casos de Testes"
TRAIN_PATH =  DATA_DIR / "Desenvolvimento da Metodologia"
