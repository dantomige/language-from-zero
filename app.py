import gradio as gr
from inference import Inference

class ModelInterface:

    def __init__(self, inference_model: Inference):
        self.inference_model = inference_model

    def predict(self, query, max_response_tokens=100):
        return self.inference_model.response(
            query, max_response_tokens=max_response_tokens
        )


def main():
    CHECKPOINT_DIR = "src/checkpoints/"
    EXPERIMENTS_FOLDER_NAME = "head"
    
    inference_model = Inference.from_experiment(
        checkpoint_dir=CHECKPOINT_DIR, experiment_folder_name=EXPERIMENTS_FOLDER_NAME
    )

    model_interface = ModelInterface(inference_model=inference_model)

    interface = gr.Interface(fn=model_interface.predict, inputs="text", outputs="text")
    interface.launch()


if __name__ == "__main__":
    main()


