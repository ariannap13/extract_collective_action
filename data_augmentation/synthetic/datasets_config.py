from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Dataset config class."""

    name: str
    augmentation_system_prompt: str
    augmentation_task_prompt: str
    classification_system_prompt: str
    classification_task_prompt: str


AdditionalDimConfig = DatasetConfig(
    name="additional_dim_augment",
    augmentation_system_prompt="You are an advanced AI writer. Your job is to help write examples of social media comments that convey certain dimensions of engagement in collective action, starting from an anchor example.",
    augmentation_task_prompt="""The following anchor example conveys the dimension {label}. {label} is defined by {social_dimension_description}. Write 20 new examples that are semantically similar to the anchor example, maintaining the same content structure but slightly varying the topic. Ensure each generated comment retains the core meaning, intent, and key details of the anchor, while introducing minor variations in topic, wording or minor shifts in context. Do not drastically change the main idea or subject matter. Put each generated example on a new line.""",
    classification_system_prompt="",
    classification_task_prompt="",
)