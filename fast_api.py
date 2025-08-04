import os
import shutil
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from enum import Enum
from loguru import logger
from typing import List

os.makedirs("logs", exist_ok=True)
logger.add("logs/fastapi_app.log",
           rotation="1 MB", retention="10 days")

"""3. Next, we specify the path to our serialized model and define a function to load the model:"""

MODEL_PATH = "models/stack_class_pipe.joblib"
def load_model(path=MODEL_PATH):
  """Load the ML model from disk."""
  try:
    model = joblib.load(path)
    logger.success(f"Model loaded successfully from {path}")
    return model
  except Exception as e:
    logger.error(f"Error loading model from {path}: {e}")
    raise

"""4. We will load the model by just accessing the function we have just created in Step 3:"""

model = load_model()

"""5. Now we start to build the FastAPI app, by initializing the app:"""

app = FastAPI(
    title="FastAPI Churn Prediction API",
    version="1.0.0",
    description="Predict customer churn risk. Supports batch, model versioning, and helpful info endpoints.")

"""6. In this step, we will use Pydantic objects for the first time. Pydantic allows you to define strict enumerated lists using Python’s **Enum**, making your data models both cleaner and more reliable. This is particularly useful for fields with fixed, limited options, such as gender, contract types, or payment methods, as shown in the following step:"""

class GenderEnum(str, Enum):
  Male = 'Male'
  Female = 'Female'

class ContractTypeEnum(str, Enum):
  MonthToMonth = 'Month-to-Month'
  OneYear = 'One-Year'
  TwoYear = 'Two-Year'

class PaymentMethodEnum(str, Enum):
  CreditCard = 'Credit Card'
  BankTransfer = 'Bank Transfer'
  ElectronicCheck = 'Electronic Check'
  MailedCheck = 'Mailed Check'

"""7. In this step, we define a **CustomerInput** schema using Pydantic's **BaseModel**, which enables automatic and reliable data validation. By specifying data types, using enumerated fields, and adding example values, we ensure that only well-structured and valid data is accepted, making the model robust, self-documenting, and safe for downstream processing:"""

class CustomerInput(BaseModel):
  CustomerID: str = Field(..., example="CUST00001")
  Age: float = Field(..., example=35)
  Gender: GenderEnum = Field(..., example='Male')
  Tenure: float = Field(..., example=22.46)
  MonthlyCharges: float = Field(..., example=86.31)
  ServiceUsage: float = Field(..., example=1.36)
  ContractType: ContractTypeEnum = Field(..., example='Month-to-Month')
  PaymentMethod: PaymentMethodEnum = Field(..., example='Credit Card')
  CustomerSupportCalls: float = Field(..., example=0.0)

class BatchInput(BaseModel):
    data: List[CustomerInput]

CHURN_RISK_MAP = {
  0: "Low Risk",
  1: "Medium Risk",
  2: "High Risk"}

"""10. In this step, we define the `make_prediction` function, which takes a DataFrame of customer data, removes the **CustomerID** column, and passes the remaining features to a preloaded model to generate predictions. It then constructs a structured result for each customer, including their original ID, the predicted churn risk label (mapped from a class index), and the associated probability scores for each risk level. The function returns a list of these results, making it easy to interpret and display model outputs:"""

def make_prediction(input_df):
    input_X = input_df.drop(columns=['CustomerID'])
    preds = model.predict(input_X)
    pred_probs = model.predict_proba(input_X)
    results = []
    for i, row in input_df.iterrows():
        pred = preds[i]
        pred_proba = pred_probs[i]
        result = {
            "CustomerID": row['CustomerID'],
            "prediction": CHURN_RISK_MAP.get(pred, "Unknown"),
            "prediction_probs": {
                "Low Risk": float(pred_proba[0]),
                "Medium Risk": float(pred_proba[1]),
                "High Risk": float(pred_proba[2])
            }
        }
        results.append(result)
    return results

"""11. As part of this step, we create a POST endpoint (namely predict) using the `@app.post("/predict", tags=["Prediction"])` decorator, which allows clients to submit a single customer's data for churn prediction (online prediction request); the function processes the input, generates a prediction with probability scores, logs the result, and returns it in a structured JSON response:"""

@app.post("/predict", tags=["Prediction"])
def predict_online(body: CustomerInput):
    """Predict churn risk for a single customer."""
    try:
        input_df = pd.DataFrame([body.dict()])
        logger.debug(f"Online prediction input: {input_df}")
        result = make_prediction(input_df)[0]
        logger.success(f"Prediction result for {body.CustomerID}: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e

"""12. To handle multiple customer predictions at once, we define the `predict_batch` function, which accepts a **BatchInput** object containing a list of customer records. It converts the input into a DataFrame, logs the batch input, and passes it to the `make_prediction` function to generate predictions for all customers. The results, including risk levels and probability scores, are returned as a list, while any errors are caught and returned as a **400 HTTP response** with detailed logging:"""
@app.post("/batch_predict", tags=["Prediction"])
def predict_batch(body: BatchInput):
    try:
        input_df = pd.DataFrame([item.dict() for item in body.data])
        logger.debug(f"Batch prediction input: {input_df}")
        results = make_prediction(input_df)
        logger.success(f"Batch prediction completed. {len(results)} results returned.")
        return results
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e

"""13. To support dynamic model updates without restarting the service, we define a PUT endpoint at `/model` using the `@app.put("/model", tags=["Model Versioning"])` decorator. This endpoint accepts an uploaded `.joblib` file containing a trained model and updates the in-memory model used for predictions. The function is marked as async, allowing it to handle the file upload operation asynchronously, meaning that the server can continue processing other requests while waiting for the file to be received. This improves performance and responsiveness, especially when handling large files or multiple simultaneous upload requests:"""

@app.put("/model", tags=["Model Versioning"])
async def update_model(file: UploadFile):
    global model
    logger.info(f"Received request to update model with file: {file.filename}")
    try:
        model_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(model_dir, exist_ok=True)
        tmp_path = os.path.join(model_dir, "tmp_model.joblib")
        with open(tmp_path, "wb") as temp_buffer:
            shutil.copyfileobj(file.file, temp_buffer)
        logger.debug(f"Model file saved temporarily to {tmp_path}")
        ph_model = joblib.load(tmp_path)
        model = ph_model
        shutil.move(tmp_path, MODEL_PATH)
        logger.success(f"Model updated successfully to {MODEL_PATH}")
        return {"message": "Model updated successfully", "model_path": MODEL_PATH}
    except (OSError, IOError) as e:
        logger.error(f"File handling error: {e}")
        raise HTTPException(status_code=500, detail=f"File handling error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error updating model: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error during model update.")

"""This step implements the same health function we used in the Flask endpoint.

14. To provide visibility into the currently deployed model, we define a GET endpoint at `/model_info` using the `@app.get("/model_info", tags=["Model Versioning"])` decorator. This function retrieves the file modification time of the model from the filesystem and returns it along with the model's file path. This allows users or developers to check when the model was last updated, which is useful for version tracking, auditing, and ensuring the correct model is in use. If an error occurs while accessing the model file, a **500 HTTP response** is returned with a detailed error message:
"""

@app.get("/model_info", tags=["Model Versioning"])
def model_info():
    try:
        mod_time = os.path.getmtime(MODEL_PATH)
        logger.info("Model info endpoint called")
        return {
            "model_path": MODEL_PATH,
            "last_updated": pd.to_datetime(mod_time, unit='s').isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""15. To make the API more self-documenting and user-friendly, we define a GET endpoint at `/features` using the `@app.get("/features", tags=["Docs"])` decorator. This endpoint returns the input schema defined in the CustomerInput model by extracting its properties from the Pydantic schema. It provides a clear and structured list of the expected input fields and their data types for the prediction endpoint, helping developers understand how to format their requests without needing to manually refer to external documentation:"""

@app.get("/features", tags=["Docs"])
def features():
    logger.info("Features endpoint called")
    return CustomerInput.schema()["properties"]

"""16. Monitoring is the next step, and to support basic monitoring and health checks, we define a GET endpoint at `/metrics` using the `@app.get("/metrics", tags=["Health"])` decorator. This function returns simple metadata about the currently deployed model file, including its size in bytes and the last time it was modified. It uses `os.stat()` to access the file's statistics and converts the modification time into a human-readable ISO timestamp. This information is useful for verifying that the model is loaded correctly, tracking updates, and integrating with external monitoring tools. If the file cannot be accessed, the function logs the error and returns a **500 HTTP** response with a relevant error message:"""

@app.get("/metrics", tags=["Health"])
def metrics():
    try:
        stat = os.stat(MODEL_PATH)
        logger.info("Metrics endpoint called")
        return {
            "model_file_size_bytes": stat.st_size,
            "last_updated": pd.to_datetime(stat.st_mtime, unit='s').isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve metrics")

"""17. The final step is to create a message when the API is accessed via a web browser, this will give an introductory message on the root and give information about how to obtain the API docs for the endpoint:"""

@app.get("/", tags=["Docs"])
def root():
    return {
        "message": "Welcome to the FastAPI Churn Prediction API!",
        "docs": "/docs",
        "redoc": "/redoc"
    }

"""In the following section, we will explore how to run the API using a serving framework called **Uvicorn**. Unlike Flask, FastAPI applications are not typically run directly from Python scripts and instead require an ASGI server like Uvicorn for proper deployment."""

