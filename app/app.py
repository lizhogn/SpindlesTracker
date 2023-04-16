
import gradio as gr
import glob
from demo import demo_app
from module.detection.onnx_inference import YoloxInference
yolox_infer = YoloxInference()

def predict(inp, conf):
    outputs = demo_app.predict(inp, conf)
    return outputs

examples = [[x, 0.2] for x in glob.glob("app/examples/*.PNG")]

demo = gr.Interface(fn=predict,
                    inputs=[gr.Image(type="numpy").style(height="auto", width="auto"),
                            gr.Slider(0, 1, value=0.2)],
                    outputs=gr.Image(type="numpy").style(height="auto", width="auto"),
                    examples=examples,
                    title="Spindle Skeleton Extraction",
                    description="YOLOX-SP + Skeletonize")

demo.launch()