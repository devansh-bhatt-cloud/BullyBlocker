from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_code import predict_message

app = FastAPI()

class Message(BaseModel):
    message: str
    
# Allow frontend access (JS fetch calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static & template folders
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# JSON input schema
class MessageInput(BaseModel):
    message: str

# Webpage route (your frontend)
@app.get("/", response_class=HTMLResponse)
def load_home(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

# Backend logic

@app.post("/predict")
async def get_prediction(data: Message):
    tag, response = predict_message(data.message)
    print(f"[DEBUG] TAG: {tag}, RESPONSE: {response}")  # 👈 Add this line
    return {
        "tag": tag,
        "response": response
    }

