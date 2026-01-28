from models.model1 import build_model1
from models.model2 import build_model2
from models.model3 import build_model3

MODELS_REGISTRY = {
    'model1': build_model1,
    'model2': build_model2,
    'model3': build_model3,
}