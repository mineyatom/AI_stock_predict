from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



def create_xgb():

    return XGBClassifier(

        n_estimators=150,

        learning_rate=0.05,

        max_depth=5,

        random_state=42,

        eval_metric="logloss"

    )



def create_rf():

    return RandomForestClassifier(

        n_estimators=300,

        max_depth=8,

        random_state=42,

        n_jobs=-1

    )



def create_lr():

    return LogisticRegression(

        max_iter=3000,

        random_state=42

    )



def create_models():

    return {

        "XGBoost":

            create_xgb(),

        "RandomForest":

            create_rf(),

        "LogisticRegression":

            create_lr()

    }