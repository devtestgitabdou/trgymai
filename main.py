from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Gym Form Detector API", version="1.0.0")

# CORS middleware - allow Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your Flutter app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    logger.info("Loading model from gym_form_detector-1.pkl...")
    with open("gym_form_detector-1.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully!")
    
    # Check model type and expected input shape
    logger.info(f"Model type: {type(model)}")
    
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Define input data model
class ExerciseFeatures(BaseModel):
    # Adjust these based on your actual model input features
    # For now, creating 3 sample features - adjust to match your model's expected input
    feature1: float = 0.0
    feature2: float = 0.0
    feature3: float = 0.0
    # Add more features as needed based on your model

class BatchPredictionRequest(BaseModel):
    data: list[list[float]]

@app.get("/")
async def root():
    return {
        "message": "Gym Form Detector API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Make a single prediction (POST)",
            "/predict-batch": "Make batch predictions (POST)"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": str(type(model))
    }

@app.post("/predict")
async def predict_single(data: ExerciseFeatures):
    """
    Make a single prediction
    """
    try:
        # Convert input to numpy array
        # Adjust feature names based on your actual model
        input_features = np.array([[
            data.feature1,
            data.feature2,
            data.feature3
            # Add more features here
        ]])
        
        logger.info(f"Input shape: {input_features.shape}")
        
        # Make prediction
        prediction = model.predict(input_features)
        
        # If it's a classifier, you might want probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_features)
            return {
                "prediction": prediction.tolist()[0],
                "probabilities": probabilities.tolist()[0] if hasattr(probabilities[0], '__len__') else probabilities.tolist(),
                "confidence": float(max(probabilities[0])) if hasattr(probabilities[0], '__len__') else float(probabilities[0])
            }
        
        return {
            "prediction": prediction.tolist()[0] if hasattr(prediction, '__len__') else float(prediction)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions
    """
    try:
        # Convert to numpy array
        input_data = np.array(request.data)
        logger.info(f"Batch input shape: {input_data.shape}")
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Get probabilities if available
        results = []
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data)
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "index": i,
                    "prediction": float(pred) if not hasattr(pred, '__len__') else pred.tolist(),
                    "probabilities": prob.tolist(),
                    "confidence": float(max(prob)) if hasattr(prob, '__len__') else float(prob)
                })
        else:
            for i, pred in enumerate(predictions):
                results.append({
                    "index": i,
                    "prediction": float(pred) if not hasattr(pred, '__len__') else pred.tolist()
                })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """
    Get information about the loaded model
    """
    try:
        info = {
            "model_type": str(type(model)),
            "has_predict_proba": hasattr(model, 'predict_proba'),
        }
        
        # Try to get model parameters
        if hasattr(model, 'get_params'):
            info["params"] = model.get_params()
        
        # Try to get feature names if available
        if hasattr(model, 'feature_names_in_'):
            info["feature_names"] = model.feature_names_in_.tolist()
        
        return info
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)