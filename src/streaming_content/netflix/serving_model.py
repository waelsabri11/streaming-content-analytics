"""MLflow pyfunc wrapper (models-from-code) so the registered model accepts raw feature rows.

The LightGBM booster needs pandas categoricals with the training categories; the serving container only gets JSON,
so the wrapper rebuilds them from the stored category lists. Logged by `pipeline.log_mlflow`.
"""

import json

import lightgbm as lgb
import mlflow
import pandas as pd


class ScoreModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.booster = lgb.Booster(model_file=context.artifacts["lgbm"])
        spec = json.load(open(context.artifacts["spec"]))
        self.cols, self.cats = spec["cols"], spec["cats"]

    def predict(self, context, model_input, params=None):
        x = pd.DataFrame(model_input)[self.cols].copy()
        for c in x.columns:
            if c in self.cats:
                x[c] = pd.Categorical(x[c], categories=self.cats[c])
            else:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        return self.booster.predict(x)  # predicted log(views in release semester)


mlflow.models.set_model(ScoreModel())
