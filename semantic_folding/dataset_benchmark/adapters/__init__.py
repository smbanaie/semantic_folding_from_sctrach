"""
Adapters for converting various QA datasets to MuSiQue-like JSONL format.
Each adapter inherits from BaseDatasetAdapter.
"""

from .base_adapter import BaseDatasetAdapter
from .pubmedqa_adapter import PubMedQAAdapter
from .belebele_adapter import BelebeleAdapter
from .bioasq_adapter import BioASQAdapter
from .maud_adapter import MaudAdapter
from .popqa_adapter import PopQAAdapter
from .nq_rear_adapter import NQRearAdapter
from .narrativeqa_adapter import NarrativeQAAdapter
from .hotpotqa_adapter import HotpotQAAdapter
from .twowiki_adapter import TwoWikiMultihopQAAdapter
from ._stubs import (
    SciDQAAdapter,
    DropAdapter,
    MultiMedQAAdapter,
    DocFinQAAdapter,
    CuadAdapter,
    MedReadMeAdapter,
    CflueAdapter,
)


ADAPTER_REGISTRY = {
    "pubmedqa": PubMedQAAdapter,
    "scidqa": SciDQAAdapter,
    "drop": DropAdapter,
    "belebele": BelebeleAdapter,
    "bioasq": BioASQAdapter,
    "multimedqa": MultiMedQAAdapter,
    "docfinqa": DocFinQAAdapter,
    "cuad": CuadAdapter,
    "maud": MaudAdapter,
    "medreadme": MedReadMeAdapter,
    "cflue": CflueAdapter,
    "popqa": PopQAAdapter,
    "nq_rear": NQRearAdapter,
    "narrativeqa": NarrativeQAAdapter,
    "hotpotqa": HotpotQAAdapter,
    "2wikimultihopqa": TwoWikiMultihopQAAdapter,
}


def get_adapter(name: str, **kwargs) -> BaseDatasetAdapter:
    """Factory: instantiate the right adapter by dataset name."""
    if name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(ADAPTER_REGISTRY)}"
        )
    return ADAPTER_REGISTRY[name](**kwargs)


__all__ = [
    "BaseDatasetAdapter",
    "get_adapter",
    "ADAPTER_REGISTRY",
    "PubMedQAAdapter",
    "SciDQAAdapter",
    "DropAdapter",
    "BelebeleAdapter",
    "BioASQAdapter",
    "MultiMedQAAdapter",
    "DocFinQAAdapter",
    "CuadAdapter",
    "MaudAdapter",
    "MedReadMeAdapter",
    "CflueAdapter",
    "PopQAAdapter",
    "NQRearAdapter",
    "NarrativeQAAdapter",
    "HotpotQAAdapter",
    "TwoWikiMultihopQAAdapter",
]
