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
    
    )

from market_price import(
    get_stock_price
)

from scheduler import start_scheduler

from ollama_analyzer import analyze_prediction_with_ollama

from feature_description import explain_feature_list

from confidence_manager import get_confidence_level
from market_summary import generate_market_summary

from summary_manager import (
    get_today_prediction_summary,
    get_market_signal,
    get_model_confidence
)

from stock_chat import chat_with_model

from model_evaluator import (
    evaluate_model,
    evaluate_confidence_bins,
    evaluate_stock_accuracy
)

from evaluation_summary import generate_evaluation_summary


app = FastAPI()
latest_prediction_result = None


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


    

    current_stock_price = None

    market_summary = None

    prediction_summary = get_today_prediction_summary()

    market_signal, market_signal_class = get_market_signal(
        prediction_summary
    )

    model_confidence, model_confidence_class = get_model_confidence(
        prediction_summary
    )

    try:
        market_summary = generate_market_summary(
            market_data=market_data,
            prediction_summary=prediction_summary
        )

    except Exception as e:
        print("AI 市場分析產生失敗：", e)
        market_summary = "AI 市場分析暫時無法產生。"

   

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
            
            "market_signal": market_signal,
            "market_signal_class": market_signal_class,
            "model_confidence": model_confidence,
            "model_confidence_class": model_confidence_class,
            
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
    global latest_prediction_result

    result = predict_stock(stock_id)

    confidence_level = get_confidence_level(
        result.get("confidence", 0)
    )

    result["confidence_level"] = confidence_level

    latest_prediction_result = result

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


@app.post("/chat")
def chat_model(
    stock_id: str = Form(...),
    question: str = Form(...)
):
    global latest_prediction_result

    if latest_prediction_result is None:
        return {
            "answer": "請先完成股票預測後，再使用 AI 模型助手。"
        }

    result = latest_prediction_result

    if result.get("stock_id") != stock_id:
        return {
            "answer": "目前 AI 模型助手只能解釋剛剛完成的股票預測結果。"
        }

    answer = chat_with_model(
        question=question,
        prediction_data=result
    )

    return {
        "answer": answer
    }
    


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


@app.get("/ranking")
def ranking_page(request: Request):

    stock_accuracy = evaluate_stock_accuracy()

    return templates.TemplateResponse(
        request=request,
        name="ranking.html",
        context={
            "stock_accuracy": stock_accuracy
        }
    )


@app.get("/evaluation")
def evaluation_page(request: Request):

    model_metrics = evaluate_model()

    confidence_bins = evaluate_confidence_bins()

    summary = generate_evaluation_summary(model_metrics,confidence_bins)

    return templates.TemplateResponse(
        request=request,
        name="evaluation.html",
        context={
            "model_metrics": model_metrics,
            "confidence_bins": confidence_bins,
            "evaluation_summary": summary,
        }
    )
