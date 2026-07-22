"""
benchmark_feature_sets.py

V11.1 Feature Experiment Manager

用途：

管理不同 Feature 組合

Baseline
V10 ROC Enhancement
Reduced Feature

"""



##########################################################################
# V11 Baseline Feature
##########################################################################

BASELINE_FEATURES = [

    "Volume_MA5",

    "Return",

    "Return_1",

    "Return_2",

    "Return_3",

    "K",

    "D",

    "Foreign_Investor",

    "Investment_Trust",

    "Dealer_self",

    "Dealer_Hedging",

    "Market_Return",

    "Market_RSI",

    "Market_Volatility",

    "NVDA_Return",

    "SOX_Return",

    "QQQ_Return",

]



##########################################################################
# V10 ROC Enhanced Feature
##########################################################################

ROC_FEATURES = [

    "Volume_MA5",

    "Return",

    "Return_1",

    "Return_2",

    "Return_3",

    "K",

    "D",

    "ROC",

    "Foreign_Investor",

    "Investment_Trust",

    "Dealer_self",

    "Dealer_Hedging",

    "Market_Return",

    "Market_RSI",

    "Market_Volatility",

    "NVDA_Return",

    "SOX_Return",

    "QQQ_Return",

]



##########################################################################
# Reduced Feature
##########################################################################

REDUCED_FEATURES = [

    "Volume_MA5",

    "Return",

    "Return_1",

    "Return_2",

    "Return_3",

    "K",

    "D",

    "ROC",

    "Foreign_Investor",

    "Investment_Trust",

    "Dealer_self",

    "Market_Return",

    "Market_RSI",

    "SOX_Return",

    "QQQ_Return",

]



##########################################################################
# Feature Registry
##########################################################################

FEATURE_SETS = {


    "Baseline":

        BASELINE_FEATURES,


    "ROC_Enhanced":

        ROC_FEATURES,


    "Reduced":

        REDUCED_FEATURES,

}





def get_feature_set(

    name

):

    """

    取得指定 Feature Set

    """

    if name not in FEATURE_SETS:

        raise ValueError(

            f"Unknown Feature Set: {name}"

        )


    return FEATURE_SETS[name]





def list_feature_sets():

    """

    列出所有 Feature Set

    """

    return list(

        FEATURE_SETS.keys()

    )