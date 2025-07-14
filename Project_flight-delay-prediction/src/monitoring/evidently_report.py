from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, ClassificationPerformanceTab
from evidently import ColumnMapping
import pandas as pd

def create_evidently_report(reference_data: pd.DataFrame, production_data: pd.DataFrame, output_path: str):
    column_mapping = ColumnMapping(
        target="is_delay",
        prediction="prediction",
        numerical_features=["distance", "CRSDepHour", "Month", "DayOfWeek"],
        categorical_features=["UniqueCarrier", "Origin", "Dest"]
    )

    dashboard = Dashboard(tabs=[DataDriftTab(), ClassificationPerformanceTab()])
    dashboard.calculate(reference_data, production_data, column_mapping=column_mapping)
    dashboard.save(output_path)

