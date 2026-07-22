"""AWS Lambda entry point for the DQC FastAPI API."""

from mangum import Mangum

from api.main import app

handler = Mangum(app, lifespan="off")
