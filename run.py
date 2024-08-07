from fastapi import FastAPI
import gradio as gr
from main import iface

app = FastAPI()

@app.get("/")
async def root():
    return "Gradio app is running at /gradio", 200

app = gr.mount_gradio_app(app, iface, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
