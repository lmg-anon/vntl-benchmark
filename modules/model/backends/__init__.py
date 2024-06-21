from .kcpp_backend import KcppModel
from .lcpp_backend import LcppModel
from .ooba_backend import OobaModel
from .oai_backend import OaiModel
try:
    from .lpy_backend import LpyModel
    LPY_PRESENT = True
except ModuleNotFoundError as e:
    LPY_PRESENT = False
try:
    from .exl2_backend import EXL2Model
    EXL2_PRESENT = True
except ModuleNotFoundError as e:
    EXL2_PRESENT = False
try:
    from .tf_backend import TFModel
    TF_PRESENT = True
except ModuleNotFoundError as e:
    TF_PRESENT = False
try:
    from .unsloth_backend import UnslothModel
    UNSLOTH_PRESENT = True
except ModuleNotFoundError as e:
    UNSLOTH_PRESENT = False
try:
    from .tlservice_backend import TLServiceModel
    TRANSLATORS_PRESENT = True
except ModuleNotFoundError as e:
    TRANSLATORS_PRESENT = False
try:
    from .sugoi_backend import SugoiModel
    SUGOI_PRESENT = True
except ModuleNotFoundError as e:
    SUGOI_PRESENT = False