"""Test-only credentials. auth.py reads these at import, so they are set
before any test module imports the app. They match the Postman local
environment (api/postman/local.postman_environment.json) and the CI workflow.
"""
import os

os.environ["JWT_SECRET"] = "test-only-jwt-secret-not-for-deployment-0123456789"  # nosec B105
os.environ["API_ADMIN_PASSWORD"] = "admin123"  # nosec B105
os.environ["API_VIEWER_PASSWORD"] = "viewer123"  # nosec B105
