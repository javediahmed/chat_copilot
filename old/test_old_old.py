import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD


class AnomalyDetection:
    def __init__(self):
        self.training_data = None
        self.models = [HBOS(), KNN(), OCSVM(), ABOD()]
        self.model_names = [model.__class__.__name__ for model in self.models]
        self.selected_model = None
        self.current_date = None  # keep track of the current date

    def get_user_input(self, prompt):
        return input(prompt).strip()

    def load_data(self, filename):
        return pd.read_csv(filename)

    def generate_random_data(self, num_columns):
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp.now().floor("D")
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        data = pd.DataFrame(
            np.random.randn(len(dates), num_columns),
            columns=[f"Variable_{i}" for i in range(1, num_columns + 1)],
            index=dates,
        )
        return data

    def process_training_data_input(self):
        print("Welcome to the Anomaly Detection Test Script.")
        print("Please select the training data source:")
        print("1. Load data from a file")
        print("2. Generate random data")
        data_source = self.get_user_input("Enter your choice (1 or 2): ")

        if data_source == "1":
            filename = self.get_user_input("Enter the file path: ")
            training_data = self.load_data(filename)
        elif data_source == "2":
            num_columns = int(self.get_user_input("Enter the number of columns (variables): "))
            training_data = self.generate_random_data(num_columns)
        else:
            print("Invalid choice. Exiting the script.")
            training_data = None

        self.current_date = training_data.index.max()  # update the current date

        return training_data

    def process_test_data_input(self):
        self.current_date += pd.DateOffset(days=1)  # increment the current date
        print(f"\nPlease select the test data source for {self.current_date.date()}:")
        test_data = pd.DataFrame(columns=self.training_data.columns, index=[self.current_date])

        user_choice = self.get_user_input("1. File\n2. Simulate\nEnter your choice: ")

        if user_choice == "1":  # File
            file_path = self.get_user_input("Enter the file path: ")
            column_data = self.load_data(file_path)
            if len(column_data.columns) != len(self.training_data.columns):
                print("Invalid data dimensions. Defaulting to simulated data.")
                test_data = self.generate_simulated_data()
            else:
                test_data.loc[self.current_date] = column_data.iloc[0].values

        elif user_choice == "2":  # Simulate
            simulated_data = self.generate_simulated_data()
            test_data = test_data.combine_first(simulated_data)

        else:
            print("Invalid choice. Exiting the script.")

        return test_data

    def generate_simulated_data(self):
        num_columns = len(self.training_data.columns)
        simulated_data = pd.DataFrame(
            np.random.normal(self.training_data.mean(), self.training_data.std(), size=(1, num_columns)),
            columns=self.training_data.columns,
            index=[self.current_date],  # use the current date as the index
        )
        return simulated_data

    def select_model(self):
        print("\nSelect the anomaly detection model:")
        for i, model_name in enumerate(self.model_names):
            print(f"{i + 1}. {model_name}")

        model_choice = self.get_user_input("Enter your choice (1 to {}): ".format(len(self.models)))

        try:
            model_index = int(model_choice) - 1
            if 0 <= model_index < len(self.models):
                self.selected_model = [
                    self.models[model_index].fit(self.training_data[[col]])
                    for col in self.training_data.columns
                ]  # train a separate model for each column
                print(f"Selected model: {self.model_names[model_index]}")
            else:
                print("Invalid choice. Defaulting to the first model.")
                self.selected_model = [
                    self.models[0].fit(self.training_data[[col]])
                    for col in self.training_data.columns
                ]  # train a separate model for each column
        except ValueError:
            print("Invalid choice. Defaulting to the first model.")
            self.selected_model = [
                self.models[0].fit(self.training_data[[col]])
                for col in self.training_data.columns
            ]  # train a separate model for each column

        self.display_summary_stats()

    def display_summary_stats(self):
        summary_stats = self.training_data.describe()
        print("\nSummary Statistics of the Training Data:")
        print(summary_stats)

    def train(self):
        print("\nTraining the anomaly detection model...")
        self.display_summary_stats()

    def test(self):
        anomaly_probabilities = pd.DataFrame(columns=self.training_data.columns)

        while True:
            test_data = self.process_test_data_input()

            if test_data is None:
                break

            print("\nRunning anomaly detection on the test data...")

            for model, col in zip(self.selected_model, self.training_data.columns):
                anomaly_score = model.decision_function(test_data[[col]])
                anomaly_probability = 1 - model.predict_proba(test_data[[col]])[0, 1]  # get the probability of the observation being an outlier
                anomaly_probabilities.loc[test_data.index[0], col] = anomaly_probability

            # print the anomaly probabilities for the current test data
            print("\nAnomaly Probabilities for Current Test Data:")
            print(anomaly_probabilities.loc[test_data.index[0]])

        print("\nFinal Anomaly Detection Results:")
        print(anomaly_probabilities)


if __name__ == "__main__":
    anomaly_detection = AnomalyDetection()
    anomaly_detection.training_data = anomaly_detection.process_training_data_input()

    if anomaly_detection.training_data is not None:
        anomaly_detection.select_model()
        anomaly_detection.train()
        anomaly_detection.test()
