"""AWS Lambda entry point for the DQC FastAPI API.

API Gateway invokes this adapter; the Angular application remains local and
uses its development proxy to call the resulting HTTPS API.
"""

from mangum import Mangum

from api.main import app

handler = Mangum(app, lifespan="off")
