from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


from predictor import predict_stock

from log_manager import(
    get_prediction_history
)

app = FastAPI()



app.mount(
    "/static",
    StaticFiles(
        directory="static"
    ),
    name="static"
)


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

@app.get("/history")
def history_page(
    request: Request
):

    data = (
    get_prediction_history()
    )   

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
           "history": data["history"],
            "accuracy": data["accuracy"],
            "validated_count":
                data["validated_count"],
            "total_count":
                data["total_count"]
        }
    )