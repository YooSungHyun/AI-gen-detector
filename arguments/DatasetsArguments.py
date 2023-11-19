from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetsArguments:
    train_datasets_path: Optional[str] = field(default="")
    eval_datasets_path: Optional[str] = field(default="")
    test_datasets_path: Optional[str] = field(default="")
    result_csv_path: Optional[str] = field(default="")
    submission_csv_path: Optional[str] = field(default="data/sample_submission.csv")
