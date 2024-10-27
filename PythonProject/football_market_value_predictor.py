import sys

import pandas as pd
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QFormLayout, QLineEdit, QMessageBox
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import statsmodels.formula.api as smf
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QComboBox, QPushButton, QWidget
from PyQt6.QtGui import QPainter, QLinearGradient
from PyQt6.QtWidgets import QMainWindow, QFormLayout, QLineEdit, QPushButton, QWidget, QMessageBox, QHBoxLayout
from PyQt6.QtCore import Qt
import numpy as np


# Load and train models
def train_models():
    global ols_model, decision_tree_model, random_forest_model, random_forest_model_def

    # Load datasets
    df_goalkeepers = pd.read_csv('goalkeepers.csv')
    df_midfielders = pd.read_csv('midfielders.csv')
    df_forwards = pd.read_csv('forwards.csv')
    df_defenders = pd.read_csv('defenders.csv')

    # Split datasets into features and target variable
    X_goalkeepers = df_goalkeepers[features_goalkeepers]
    df_goalkeepers['ln_value'] = np.log(df_goalkeepers['value'])
    y_goalkeepers = df_goalkeepers['ln_value']

    # X_midfielders = df_midfielders[features_midfielders]
    df_midfielders['ln_value'] = np.log(df_midfielders['value'])
    # y_midfielders = df_midfielders['ln_value']

    X_forwards = df_forwards[features_forwards]
    df_forwards['ln_value'] = np.log(df_forwards['value'])
    y_forwards = df_forwards['ln_value']

    X_defenders = df_defenders[features_defenders]
    df_defenders['ln_value'] = np.log(df_defenders['value'])
    y_defenders = df_defenders['ln_value']

    # Train models
    formula = 'ln_value ~ ' + ' + '.join(features_goalkeepers)
    data_gk = pd.concat([X_goalkeepers, y_goalkeepers], axis=1)
    ols_model = smf.ols(formula, data=data_gk).fit()

    decision_tree_model = DecisionTreeRegressor(random_state=42)
    decision_tree_model.fit(df_midfielders[features_midfielders], df_midfielders['ln_value'])
    random_forest_model = RandomForestRegressor().fit(X_forwards, y_forwards)
    random_forest_model_def = RandomForestRegressor().fit(X_defenders, y_defenders)


# Feature sets
features_midfielders = [
    'age', 'goals', 'CL', 'passes_completed_short', 'passes_into_final_third',
    'Pts', 'xG', 'xGA', 'xg_xa_per90', 'carry_distance', 'tackles_won',
    'isPremierLeague', 'isLigue1', 'isSerieA', 'isLaLiga', 'isBundesliga'
]
features_defenders = [
    'age', 'CL', 'goals', 'xg_xa_per90', 'passes_ground', 'touches_att_pen_area',
    'touches_def_pen_area', 'aerials_won_pct', 'Pts', 'xGA', 'xG',
    'isPremierLeague', 'isLigue1', 'isSerieA', 'isLaLiga', 'isBundesliga'
]
features_forwards = [
    'age', 'CL', 'goals', 'gca', 'Pts', 'xG', 'xGA', 'dribbles_completed',
    'xg_xa_per90', 'touches_att_pen_area', 'passes_into_final_third',
    'isPremierLeague', 'isLigue1', 'isSerieA', 'isLaLiga', 'isBundesliga'
]
features_goalkeepers = [
    'age', 'CL', 'wins_gk', 'draws_gk', 'passes_pct_launched_gk',
    'psnpxg_per_shot_on_target_against', 'clean_sheets', 'isPremierLeague', 'isLigue1',
    'isLaLiga', 'isSerieA', 'isBundesliga'
]

# Call this function to train the models
train_models()

from PyQt6.QtWidgets import QMainWindow, QLabel
from PyQt6.QtGui import QPixmap


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Football Player Value Predictor")
        self.setGeometry(100, 100, 800, 600)  # Adjust the size as needed

        # Create a central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Add a QLabel for the background image
        self.background_label = QLabel(self)
        self.background_label.setPixmap(QPixmap('background.jpg'))
        self.background_label.setScaledContents(True)

        # Set the central widget and layout
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)

        # Add your start button with custom size
        self.start_button = QPushButton("Start")
        self.start_button.setFixedHeight(60)  # Adjust height as needed
        self.start_button.setFixedWidth(150)  # Adjust width as needed
        self.start_button.clicked.connect(self.show_position_selection)

        # Add the button to the layout with center alignment
        layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Ensure the layout stretches to center the button
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def paintEvent(self, event):
        super().paintEvent(event)
        # Use QPainter to paint the background image
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.background_label.pixmap())

    def show_position_selection(self):
        self.position_window = PositionSelectionWindow()
        self.position_window.show()
        self.close()


class PositionSelectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Player Position")
        self.setGeometry(100, 100, 800, 600)  # Adjust the size to match the main window

        # Create a central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)

        # Add a combo box for selecting position
        self.position_combo = QComboBox()
        self.position_combo.addItems(["Goalkeeper", "Midfielder", "Forward", "Defender"])
        self.position_combo.setFixedWidth(300)  # Set a fixed width for the combo box
        self.position_combo.setFixedHeight(40)  # Set a fixed height for the combo box

        self.layout.addWidget(self.position_combo, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Add select button
        self.select_button = QPushButton("Select")
        self.select_button.clicked.connect(self.show_position_form)
        self.select_button.setFixedHeight(40)  # Adjust height as needed
        self.select_button.setFixedWidth(150)  # Adjust width as needed
        self.select_button.setStyleSheet(
            "background-color: #003366;"  # Darker blue for the button
            "color: white;"
            "border: none;"
            "border-radius: 10px;"
            "font-size: 16px;"
            "padding: 10px;"
            "min-width: 150px;"
        )

        self.layout.addWidget(self.select_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Ensure the layout stretches to center the widgets
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        # Create a background gradient from black to dark blue
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(0, 0, 0))  # Black
        gradient.setColorAt(1, QColor(0, 0, 50))  # Dark blue

        painter.fillRect(self.rect(), gradient)

    def show_position_form(self):
        position = self.position_combo.currentText()
        if position == "Goalkeeper":
            self.form_window = PositionFormWindow("Goalkeeper", features_goalkeepers, ols_model)
        elif position == "Midfielder":
            self.form_window = PositionFormWindow("Midfielder", features_midfielders, decision_tree_model)
        elif position == "Forward":
            self.form_window = PositionFormWindow("Forward", features_forwards, random_forest_model)
        elif position == "Defender":
            self.form_window = PositionFormWindow("Defender", features_defenders, random_forest_model_def)
        self.form_window.show()
        self.close()


class PositionFormWindow(QMainWindow):
    def __init__(self, position, features, model):
        super().__init__()
        self.setWindowTitle(f"{position} Features Input")
        self.features = features
        self.model = model
        self.inputs = {}

        # Set the size of the window to match other pages
        self.setGeometry(100, 100, 800, 600)

        # Create a main layout for the window
        main_layout = QVBoxLayout()

        # Add the back button at the upper left corner
        self.back_button = QPushButton("Back")
        self.back_button.setFixedHeight(40)
        self.back_button.setFixedWidth(80)
        self.back_button.clicked.connect(self.go_back)

        # Create a horizontal layout for the back button (aligned to the left)
        back_button_layout = QHBoxLayout()
        back_button_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        back_button_layout.addWidget(self.back_button)

        # Add the back button layout to the main layout
        main_layout.addLayout(back_button_layout)

        # Create a form layout for the features input fields
        form_layout = QFormLayout()

        # Add the league selection combo box
        self.league_combo = QComboBox()
        self.league_combo.addItems(["Premier League", "Ligue 1", "Serie A", "La Liga", "Bundesliga"])
        self.league_combo.setFixedWidth(150)
        league_label = QLabel("Select League:")
        league_label.setStyleSheet("color: white;")  # Set color for this specific label
        form_layout.addRow(league_label, self.league_combo)

        for feature in features:
            if feature.startswith("is"):  # Skip league features for now
                continue
            input_field = QLineEdit()
            input_field.setFixedWidth(150)
            input_field.setToolTip(self.get_feature_description(feature))  # Set tooltip for hover description
            self.inputs[feature] = input_field
            # form_layout.addRow(self.get_feature_description(feature), input_field)
            label = QLabel(self.get_feature_description(feature))
            label.setStyleSheet("color: white;")  # Set color only for this label
            form_layout.addRow(label, input_field)

        # Add the form layout to the main layout
        main_layout.addLayout(form_layout)

        # Create a horizontal layout for the predict and exit buttons (aligned to the center)
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create and size the predict button
        self.predict_button = QPushButton("Predict Value")
        self.predict_button.setFixedHeight(60)  # Adjust height as needed
        self.predict_button.setFixedWidth(150)  # Adjust width as needed
        self.predict_button.clicked.connect(self.predict_value)
        button_layout.addWidget(self.predict_button)

        # Create and size the exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.setFixedHeight(30)  # Adjust height as needed
        self.exit_button.setFixedWidth(150)  # Adjust width as needed
        self.exit_button.clicked.connect(self.close_app)
        button_layout.addWidget(self.exit_button)

        # Add the button layout to the main layout
        main_layout.addLayout(button_layout)

        # Set the main layout as the central widget's layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def go_back(self):
        self.position_window = PositionSelectionWindow()
        self.position_window.show()
        self.close()

    def close_app(self):
        self.close()

    def get_feature_description(self, feature):
        # Define descriptions for each feature set
        descriptions = {
            'age': 'The player’s age.',
            'goals': 'Number of goals scored by the player.',
            'CL': 'Champions League appearances (1 for Yes, 0 for No).',
            'passes_completed_short': 'Number of short passes completed (between 5 and 15 yards).',
            'passes_into_final_third': 'Number of passes into attacking third of football pitch.',
            'Pts': 'Points earned by the player’s team.',
            'xG': 'Player’s team Expected Goals.',
            'xGA': 'Player’s team Expected Goals Against.',
            'xg_xa_per90': 'Expected Goals + Expected Assists per 90 minutes.',
            'carry_distance': 'Distance covered while carrying the ball.',
            'tackles_won': 'Number of tackles won.',
            'isPremierLeague': 'Is the player in the Premier League (1 for Yes, 0 for No).',
            'isLigue1': 'Is the player in Ligue 1 (1 for Yes, 0 for No).',
            'isSerieA': 'Is the player in Serie A (1 for Yes, 0 for No).',
            'isLaLiga': 'Is the player in La Liga (1 for Yes, 0 for No).',
            'isBundesliga': 'Is the player in Bundesliga (1 for Yes, 0 for No).',
            'wins_gk': 'Number of wins of player’s team when he played as a goalkeeper.',
            'draws_gk': 'Number of draws of player’s team when he played as a goalkeeper.',
            'passes_pct_launched_gk': 'Percentage of passes launched longer than 40 yards that were launched.',
            'psnpxg_per_shot_on_target_against': 'Post-Shot Expected Goals per shot on target against.',
            'clean_sheets': 'Number of clean sheets.',
            'passes_ground': 'Number of ground passes made by the player.',
            'touches_att_pen_area': 'Touches in the attacking penalty area.',
            'touches_def_pen_area': 'Touches in the defending penalty area.',
            'aerials_won_pct': 'Percentage of aerial duels won by the player.',
            'gca': 'Goal Creating Actions (passes leading to goals).',
            'dribbles_completed': 'Number of completed dribbles by the player.'
        }
        return descriptions.get(feature, "No description available.")

    def predict_value(self):
        input_values = []

        # Initialize league features to 0
        league_features = {
            'isPremierLeague': 0,
            'isLigue1': 0,
            'isSerieA': 0,
            'isLaLiga': 0,
            'isBundesliga': 0
        }

        # Set the selected league feature to 1
        selected_league = self.league_combo.currentText()
        if selected_league == "Premier League":
            league_features['isPremierLeague'] = 1
        elif selected_league == "Ligue 1":
            league_features['isLigue1'] = 1
        elif selected_league == "Serie A":
            league_features['isSerieA'] = 1
        elif selected_league == "La Liga":
            league_features['isLaLiga'] = 1
        elif selected_league == "Bundesliga":
            league_features['isBundesliga'] = 1

        try:
            for feature in self.features:
                if feature in league_features:
                    input_value = league_features[feature]
                else:
                    input_value = float(self.inputs[feature].text())
                input_values.append(input_value)

            input_df = pd.DataFrame([input_values], columns=self.features)
            predicted_value = self.model.predict(input_df)
            QMessageBox.information(self, "Predicted Value",
                                    f"The predicted value is: €{np.exp(predicted_value[0]):.2f}")
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values.")

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        # Create a background gradient from black to dark blue
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(0, 0, 0))  # Black
        gradient.setColorAt(1, QColor(0, 0, 50))  # Dark blue

        painter.fillRect(self.rect(), gradient)


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
