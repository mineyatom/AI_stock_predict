from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

from predictor import predict_stock

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/predict")
def predict_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "result": None
        }
    )


@app.post("/predict")
def run_predict(
        request: Request,
        stock_id: str = Form(...)
):

    result = predict_stock(stock_id)

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "result": result
        }
    )