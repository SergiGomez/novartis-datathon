
def make_prediction(model, test_X):

    test_y_pred = model.predict(test_X, num_iteration = model.best_iteration)

    return test_y_pred


def prepare_submission(test_Y):

    return test_Y
