from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Dataset config class."""

    name: str
    augmentation_system_prompt: str
    augmentation_task_prompt: str
    classification_system_prompt: str
    classification_task_prompt: str


LabelsDef = DatasetConfig(
    name="labels_def",
    augmentation_system_prompt="",
    augmentation_task_prompt="""""",
    classification_system_prompt="You are an advanced re-writing AI. You are tasked with re-writing definitions of collective action dimensions to make them easier to understand for a Large Language Model during zero-shot tasks.",
    classification_task_prompt="You have the following knowledge about collective action dimensions that can be expressed in social media comments: {dimdef}. Re-write the definition for each of the dimensions in {labels_dims}. Answer ONLY with 'dimension':'definition'.",
)