from transformers import pipeline

#model = pipeline("text-generation")


def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion

import gradio as gr

#gr.Interface(fn=predict, inputs="text", outputs="text").launch(share=True)
from transformers import pipeline
import gradio as gr

model = pipeline("automatic-speech-recognition")


def transcribe_audio(audio):
    transcription = model(audio)["text"]
    return transcription


gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
).launch()