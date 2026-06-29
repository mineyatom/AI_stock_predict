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

from feature_description import explain_feature_list

from confidence_manager import get_confidence_level
from market_summary import generate_market_summary


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

    market_data = get_market_data()

    history_data = get_prediction_history()

    latest_prediction = get_latest_prediction()

    accuracy_chart = get_accuracy_chart_data()

    confidence_stats = get_confidence_stats()

    recent_accuracy_stats = get_recent_accuracy_stats()

    high_confidence_accuracy = get_high_confidence_accuracy()

    stock_accuracy_stats = get_stock_accuracy_stats()

    current_stock_price = None

    market_summary = None

    try:
       

        prediction_summary = {
            "total_stocks": 0,
            "up_count": 0,
            "down_count": 0,
            "high_confidence_count": 0,
            "medium_confidence_count": 0,
            "low_confidence_count": 0,
            "stocks": []
        }

        for item in confidence_stats:
            range_name = item.get("range", "")
            count = item.get("count", 0)

            if range_name == "80%以上":
                prediction_summary["high_confidence_count"] += count

            elif range_name == "70~80%":
                prediction_summary["medium_confidence_count"] += count

            else:
                prediction_summary["low_confidence_count"] += count

            prediction_summary["total_stocks"] += count

        if latest_prediction:
            prediction_summary["stocks"].append({
                "stock_id": latest_prediction.get("stock_id", ""),
                "stock_name": latest_prediction.get("stock_name", ""),
                "prediction": latest_prediction.get("prediction", ""),
                "confidence": latest_prediction.get("confidence", ""),
                "confidence_level": latest_prediction.get(
                    "confidence_level",
                    ""
                )
            })

        market_summary = generate_market_summary(
            market_data=market_data,
            prediction_summary=prediction_summary
        )

    except Exception as e:
        print("AI 每日市場摘要產生失敗：", e)
        market_summary = "AI 每日市場摘要暫時無法產生。"

    if latest_prediction:
        current_stock_price = get_stock_price(
            latest_prediction["stock_id"]
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "market_data": market_data,
            "market_summary": market_summary,
            "accuracy": history_data["accuracy"],
            "validated_count": history_data["validated_count"],
            "total_count": history_data["total_count"],
            "latest_prediction": latest_prediction,
            "accuracy_chart": accuracy_chart,
            "current_stock_price": current_stock_price,
            "confidence_stats": confidence_stats,
            "recent_accuracy_stats": recent_accuracy_stats,
            "high_confidence_accuracy": high_confidence_accuracy,
            "stock_accuracy_stats": stock_accuracy_stats,
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

    confidence_level = get_confidence_level(
        result.get("confidence", 0)
    )

    result["confidence_level"] = confidence_level

    ai_analysis = analyze_prediction_with_ollama(
        result
    )

    positive_factors = explain_feature_list(
        result.get("positive_factors", [])
    )

    negative_factors = explain_feature_list(
        result.get("negative_factors", [])
    )

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "result": result,
            "ai_analysis": ai_analysis,
            "positive_factors": positive_factors,
            "negative_factors": negative_factors,
            "confidence_level": confidence_level,
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

