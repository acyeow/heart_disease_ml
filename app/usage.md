uvicorn endpoint:app --reload
http POST :8000/predict data:='[1.0, 2.0, 3.0, 4.0, 5.0]'
