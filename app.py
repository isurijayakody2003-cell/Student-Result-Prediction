
from flask import Flask, render_template, request
from pathlib import Path
import pickle
import numpy as np

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "model.pkl"

# --------------------------------------------------
# 2. Check required files
# --------------------------------------------------

print("Project folder:", BASE_DIR)
print("Templates folder:", TEMPLATE_DIR)
print("HTML file exists:", (TEMPLATE_DIR / "index.html").is_file())
print("Model file exists:", MODEL_PATH.is_file())

# --------------------------------------------------
# 3. Create Flask application
# --------------------------------------------------

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR)
)

# --------------------------------------------------
# 4. Load trained ML model
# --------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"model.pkl not found in: {BASE_DIR}"
    )

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("Machine learning model loaded successfully!")


# --------------------------------------------------
# 5. Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# 6. Predict student result
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from HTML form
        attendance = float(request.form["attendance"])
        assignment = float(request.form["assignment"])
        quiz = float(request.form["quiz"])
        mid_exam = float(request.form["mid_exam"])

        values = [
            attendance,
            assignment,
            quiz,
            mid_exam
        ]

        # Validate marks
        if any(not 0 <= value <= 100 for value in values):
            return render_template(
                "index.html",
                error="All values must be between 0 and 100.",
                attendance=attendance,
                assignment=assignment,
                quiz=quiz,
                mid_exam=mid_exam
            )

        # Prepare input for the ML model
        input_data = np.array([values])

        # Generate prediction
        result = model.predict(input_data)[0]

        # Display prediction on the web page
        return render_template(
            "index.html",
            prediction=str(result),
            attendance=attendance,
            assignment=assignment,
            quiz=quiz,
            mid_exam=mid_exam
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid numeric values."
        )

    except Exception as error:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error="Prediction failed. Please check the model and input features."
        )


# --------------------------------------------------
# 7. Run Flask server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
