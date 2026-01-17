from fastapi import FastAPI, HTTPException, UploadFile, File
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
import cv2
import io
from PIL import Image
import base64
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Exercise Form Correction API",
    version="1.0.0",
    description="Real-time AI for correcting exercise form using camera input"
)

# CORS middleware - allow Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = None
try:
    logger.info("Loading exercise form detection model...")
    with open("gym_form_detector-1.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("✅ Model loaded successfully!")
    logger.info(f"Model type: {type(model)}")
    
    # Check model capabilities
    if hasattr(model, 'n_features_in_'):
        logger.info(f"📊 Model expects {model.n_features_in_} features")
    if hasattr(model, 'classes_'):
        logger.info(f"🏷️ Model classes: {model.classes_}")
        
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    raise

# Define Pydantic models for different input types
class KeypointsRequest(BaseModel):
    """Input for pose keypoints (from MediaPipe/MoveNet)"""
    keypoints: list[list[float]]  # [[x1, y1, z1, score1], ...]
    exercise_type: str = "squat"  # squat, pushup, deadlift, etc.
    timestamp: float = 0.0

class FrameAnalysisRequest(BaseModel):
    """Input for direct frame analysis"""
    image_base64: str  # Base64 encoded image
    exercise_type: str = "squat"

class ExerciseFeedback(BaseModel):
    """Output model for exercise feedback"""
    is_correct: bool
    confidence: float
    corrections: list[str]
    score: float
    keypoints_confidence: dict[str, float]
    suggested_improvements: list[str]

# Helper functions
def preprocess_keypoints(keypoints: list[list[float]]) -> np.ndarray:
    """Convert keypoints to model input format"""
    # Flatten keypoints array
    flat_keypoints = np.array(keypoints).flatten()
    
    # Reshape for model (1 sample, n_features)
    return flat_keypoints.reshape(1, -1)

def analyze_form_quality(keypoints, exercise_type: str) -> dict:
    """Analyze exercise form quality based on keypoints"""
    # This is where your model makes predictions
    input_data = preprocess_keypoints(keypoints)
    
    # Make prediction
    if hasattr(model, 'predict_proba'):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(max(probabilities))
    else:
        prediction = model.predict(input_data)[0]
        confidence = 1.0
    
    # Generate feedback based on exercise type
    feedback = generate_exercise_feedback(prediction, exercise_type, keypoints)
    
    return {
        "is_correct": bool(prediction == 1),  # Assuming 1 = correct form
        "confidence": confidence,
        "prediction": int(prediction),
        "feedback": feedback
    }

def generate_exercise_feedback(prediction: int, exercise_type: str, keypoints: list) -> list[str]:
    """Generate specific feedback for different exercises"""
    feedback = []
    
    if prediction == 0:  # Incorrect form
        if exercise_type.lower() == "squat":
            feedback = [
                "Keep your back straight",
                "Go deeper in your squat",
                "Knees should not go past toes"
            ]
        elif exercise_type.lower() == "pushup":
            feedback = [
                "Keep your body in a straight line",
                "Lower your chest closer to the ground",
                "Engage your core muscles"
            ]
        elif exercise_type.lower() == "deadlift":
            feedback = [
                "Keep your back neutral",
                "Drive through your heels",
                "Engage your lats before lifting"
            ]
        else:
            feedback = ["Check your form alignment"]
    
    return feedback[:3]  # Return top 3 corrections

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Exercise Form Correction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "/": "API info",
            "/health": "Health check",
            "/model-info": "Model information",
            "/analyze/keypoints": "Analyze pose keypoints (POST)",
            "/analyze/frame": "Analyze image frame (POST)",
            "/exercises": "Get supported exercises"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model else None,
        "timestamp": np.datetime64('now').astype(str)
    }

@app.get("/model-info")
async def model_info():
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    info = {
        "model_type": str(type(model)),
        "has_predict_proba": hasattr(model, 'predict_proba'),
    }
    
    # Add model-specific info
    if hasattr(model, 'n_features_in_'):
        info["n_features_in"] = int(model.n_features_in_)
        info["expected_keypoints"] = int(model.n_features_in_ / 4)  # Assuming x,y,z,score per keypoint
    
    if hasattr(model, 'classes_'):
        info["classes"] = model.classes_.tolist()
    
    # Get parameters if available
    if hasattr(model, 'get_params'):
        try:
            params = model.get_params()
            # Convert numpy types
            for key, value in params.items():
                if hasattr(value, 'item'):
                    params[key] = value.item()
                elif isinstance(value, np.ndarray):
                    params[key] = value.tolist()
            info["params"] = params
        except:
            pass
    
    return info

@app.get("/exercises")
async def get_supported_exercises():
    """Get list of supported exercises"""
    return {
        "supported_exercises": [
            "squat", "pushup", "deadlift", "lunge", 
            "bench_press", "shoulder_press", "bicep_curl",
            "tricep_extension", "plank", "pullup"
        ],
        "default": "squat"
    }

@app.post("/analyze/keypoints")
async def analyze_pose_keypoints(request: KeypointsRequest):
    """
    Analyze exercise form from pose keypoints
    
    Example request:
    {
        "keypoints": [[0.1, 0.2, 0.3, 0.9], [0.4, 0.5, 0.6, 0.8], ...],
        "exercise_type": "squat",
        "timestamp": 1234.56
    }
    """
    try:
        logger.info(f"Analyzing {request.exercise_type} at timestamp {request.timestamp}")
        logger.info(f"Received {len(request.keypoints)} keypoints")
        
        # Validate input
        if not request.keypoints:
            raise HTTPException(status_code=400, detail="No keypoints provided")
        
        # Analyze form
        result = analyze_form_quality(request.keypoints, request.exercise_type)
        
        # Add additional metrics
        result.update({
            "exercise_type": request.exercise_type,
            "timestamp": request.timestamp,
            "keypoints_count": len(request.keypoints),
            "analysis_time": np.datetime64('now').astype(str)
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Keypoints analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/frame")
async def analyze_image_frame(
    exercise_type: str = "squat",
    file: UploadFile = File(...)
):
    """
    Analyze exercise form from image frame
    
    Note: This endpoint expects the model to work with images directly
    If your model expects keypoints, use /analyze/keypoints instead
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        logger.info(f"Analyzing {exercise_type} from image: {file.filename}")
        logger.info(f"Image shape: {image.shape}")
        
        # Here you would process the image for your model
        # Since your model likely expects keypoints, you might need to:
        # 1. Extract pose keypoints (using MediaPipe in Flutter)
        # 2. Send keypoints to /analyze/keypoints endpoint
        
        return {
            "message": "Image received. For pose analysis, extract keypoints in Flutter and use /analyze/keypoints endpoint",
            "exercise_type": exercise_type,
            "image_size": f"{image.shape[1]}x{image.shape[0]}",
            "channels": image.shape[2],
            "suggestion": "Use pose detection in Flutter, then send keypoints to /analyze/keypoints"
        }
        
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def analyze_batch_keypoints(requests: list[KeypointsRequest]):
    """
    Analyze multiple frames at once (for video analysis)
    """
    try:
        results = []
        for i, request in enumerate(requests):
            result = analyze_form_quality(request.keypoints, request.exercise_type)
            result.update({
                "frame_index": i,
                "exercise_type": request.exercise_type,
                "timestamp": request.timestamp
            })
            results.append(result)
        
        # Calculate overall form score
        correct_frames = sum(1 for r in results if r["is_correct"])
        overall_score = correct_frames / len(results) if results else 0
        
        return {
            "results": results,
            "summary": {
                "total_frames": len(results),
                "correct_frames": correct_frames,
                "incorrect_frames": len(results) - correct_frames,
                "overall_score": overall_score,
                "average_confidence": np.mean([r["confidence"] for r in results]) if results else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time WebSocket support (optional)
from fastapi import WebSocket

@app.websocket("/ws/analyze")
async def websocket_analysis(websocket: WebSocket):
    """WebSocket for real-time analysis"""
    await websocket.accept()
    try:
        while True:
            # Receive keypoints data from Flutter
            data = await websocket.receive_json()
            
            # Process the data
            keypoints = data.get("keypoints", [])
            exercise_type = data.get("exercise_type", "squat")
            
            # Analyze form
            result = analyze_form_quality(keypoints, exercise_type)
            
            # Send feedback back to Flutter
            await websocket.send_json({
                "feedback": result["feedback"],
                "is_correct": result["is_correct"],
                "confidence": result["confidence"],
                "timestamp": data.get("timestamp", 0)
            })
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )