import gradio as gr
def set_small_local():
    return (gr.Slider.update(value=25), gr.Slider.update(value=0.15), gr.Slider.update(value=1), gr.Slider.update(value=4))
def set_major_local():
    return (gr.Slider.update(value=25), gr.Slider.update(value=0.25), gr.Slider.update(value=35), gr.Slider.update(value=10))
def set_major_global():
    return (gr.Slider.update(value=30), gr.Slider.update(value=0.1), gr.Slider.update(value=2), gr.Slider.update(value=0.2))