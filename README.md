# AI305 Project – Saudi Arabia Used Cars

Price prediction (regression) and Gear_Type / Options classification on the
syarah.com used-cars dataset (8,248 ads), served by a Dockerized Flask API.

## Structure
```
data/UsedCarsSA_Unclean_EN.csv   original dataset (8,248 rows) from Kaggle
src/common.py                    loading, cleaning rules, preprocessing pipeline
src/eda.py                       Task 1 – exploration (figures + stats)
src/train_regression.py          Task 2 – LR, DT, RF, SVR, MLP + GridSearchCV
src/train_classification.py      Task 3 – LogReg, DT, RF, SVM, MLP on Gear_Type and Options
app/app.py                       Task 4 – Flask API (2 routes)
models/                          trained models (best_* are used by the API)
reports/figures, reports/results outputs of the scripts
Dockerfile, requirements.txt, test_api.py
```

## Notebooks (recommended)
Open the `notebooks/` folder in Jupyter and run each notebook top to bottom (Kernel → Restart & Run All):

| Notebook | Task |
| --- | --- |
| 01_EDA.ipynb | Task 1 – data exploration |
| 02_Regression.ipynb | Task 2 – price prediction (~5 min) |
| 03_Classification.ipynb | Task 3 – Gear_Type and Options (~5 min) |
| 04_Deployment_Test.ipynb | Task 4 – calls the running Docker API |

The notebooks import shared helpers from `src/common.py` and save models to `models/`.
They need scikit-learn 1.2 or newer. If you re-train in the notebooks, run the first code cell of
notebook 04 to write your package versions into `requirements.txt`, then rebuild the Docker image.

## Reproduce with scripts (same code as the notebooks)
```bash
pip install -r requirements.txt
python src/eda.py
python src/train_regression.py        # ~5 min
python src/train_classification.py    # ~5 min
```

## Run the API with Docker
```bash
docker build -t car-api .
docker run -d --name car-api -p 5000:5000 car-api
python test_api.py                    # calls both endpoints
docker logs car-api                   # container logs
```

Example request:
```bash
curl -X POST http://localhost:5000/predict/price -H "Content-Type: application/json" \
  -d '{"Make":"Toyota","Type":"Camry","Year":2019,"Origin":"Saudi","Color":"White","Options":"Semi Full","Engine_Size":2.5,"Fuel_Type":"Gas","Gear_Type":"Automatic","Mileage":85000,"Region":"Riyadh"}'

curl -X POST "http://localhost:5000/predict/class?target=gear_type" -H "Content-Type: application/json" \
  -d '{"Make":"Toyota","Type":"Hilux","Year":2015,"Origin":"Saudi","Color":"White","Options":"Standard","Engine_Size":2.7,"Fuel_Type":"Gas","Mileage":210000,"Region":"Riyadh","Price":45000}'
```
`/predict/class?target=options` predicts Options instead (send every field except Options).
