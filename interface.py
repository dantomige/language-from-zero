import gradio as gr


def process(user_input):
    return f"Model processed: {user_input}"

interface = gr.Interface(fn=process, inputs="text", outputs="text")

interface.launch()