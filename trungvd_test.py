from utils.missing_pay_utils import get_training_result, make_pseudo_log_classifier

if __name__ == "__main__":
    #target_curve = "RESFLAG"
    target_curve = "PAYFLAG"
    target_well = "BH-1210"
    input_curves= ["VCLAV", "PHIE", "SW"] # input features
    training_wells = ["BH-1602", "BH-25", "BH-8010"]    # training wells
    model = "random_forest"
    params = {}

    make_pseudo_log_classifier(target_curve, target_well, input_curves, training_wells, model, params)
    #result = get_training_result(target_curve, target_well, model)
    #print(result)
