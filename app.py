from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from market_data import(
    get_market_data
)

from predictor import predict_stock

from log_manager import(
    get_latest_prediction,
    get_prediction_history,
    update_prediction_result,
    get_accuracy_chart_data,
    get_confidence_stats,
    get_recent_accuracy_stats,
    get_high_confidence_accuracy,
    get_stock_accuracy_stats
    )

from market_price import(
    get_stock_price
)

from scheduler import start_scheduler

from ollama_analyzer import analyze_prediction_with_ollama


app = FastAPI()


@app.on_event("startup")
def startup_event():
    start_scheduler()


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
    update_prediction_result()

    market_data = (get_market_data())

    history_data = (get_prediction_history())

    latest_prediction = get_latest_prediction()

    accuracy_chart = get_accuracy_chart_data()

    confidence_stats =  get_confidence_stats()

    recent_accuracy_stats = get_recent_accuracy_stats()

    high_confidence_accuracy = get_high_confidence_accuracy()

    stock_accuracy_stats = (get_stock_accuracy_stats())

    current_stock_price = None

    if latest_prediction:

        current_stock_price = (
          get_stock_price(
             latest_prediction[
                "stock_id"
             ]
         )
       )
    

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "market_data":market_data,

            "accuracy":history_data["accuracy"],    

            "validated_count":history_data["validated_count"],

            "total_count":history_data["total_count"],

            "latest_prediction":latest_prediction,

            "accuracy_chart" : accuracy_chart,

            "current_stock_price": current_stock_price,

            "confidence_stats": confidence_stats,

            "recent_accuracy_stats": recent_accuracy_stats,

            "high_confidence_accuracy": high_confidence_accuracy,
            "stock_accuracy_stats":stock_accuracy_stats,

             
                       
        }
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

    ai_analysis = analyze_prediction_with_ollama(result)

 

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
        "result": result,
        "ai_analysis": ai_analysis
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

