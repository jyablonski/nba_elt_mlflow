class TrainingPipeline(Pipeline):
    """
    Class -> TrainingPipeline, ParentClass -> Sklearn-Pipeline
    Extends from Scikit-Learn Pipeline class. Additional functionality to track model metrics and log model artifacts with mlflow
    params:
    steps: list of tuple (similar to Scikit-Learn Pipeline class)
    """
    def __init__(self, steps):
        super().__init__(steps)
    
    def fit(self, X_train, y_train):
        self.__pipeline = super().fit(X_train, y_train)
        return self.__pipeline

    def get_metrics(self, y_true, y_pred, y_pred_prob):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        entropy = log_loss(y_true, y_pred_prob)
        return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}

    def make_model_name(self, experiment_name, run_name):
        clock_time = time.ctime().replace(' ', '-')
        return experiment_name + '_' + run_name + '_' + clock_time

    def log_model(self, model_key, X_test, y_test, experiment_name, run_name, run_params=None):
        model = self.__pipeline.get_params()[model_key]
        y_pred = self.__pipeline.predict(X_test)
        y_pred_prob = self.__pipeline.predict_proba(X_test)
        run_metrics = self.get_metrics(y_test, y_pred, y_pred_prob)
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri('http://localhost:5000')
        with mlflow.start_run(run_name=run_name):
            if not run_params == None:
                for name in run_params:
                    mlflow.log_param(name, run_params[name])
                for name in run_metrics:
                    mlflow.log_metric(name, run_metrics[name])
                    model_name = self.make_model_name(experiment_name, run_name)
                    mlflow.sklearn.log_model(sk_model=self.__pipeline, artifact_path='diabetes-model', registered_model_name=model_name)
                    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))
                    return run_metrics

# compare various metrics of model
def get_metrics(y_true, y_pred, y_pred_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    entropy = log_loss(y_true, y_pred_prob)
    return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}

# log model metrics with MLflow
def log_model(experiment_name, run_name, run_metrics, run_params=None):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        if not run_params == None:
            for name in run_params:
                mlflow.log_param(run_params[name])
        for name in run_metrics:
            mlflow.log_metric(name, run_metrics[name])
        
    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))

################### mlflow
model_pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), ('model', LogisticRegression())])
model_pipeline.fit(past_games, past_games_outcome)