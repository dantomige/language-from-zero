from pydantic import BaseModel

class TrainConfig(BaseModel):
    learning_rate: float
    num_epochs: int