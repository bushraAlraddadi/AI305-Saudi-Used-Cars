"""Calls both endpoints of the running container and prints the answers.
Usage:  python test_api.py            (API on http://localhost:5000)
"""
import json
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"

cars = [
    {"Make": "Toyota", "Type": "Camry", "Year": 2019, "Origin": "Saudi", "Color": "White",
     "Options": "Semi Full", "Engine_Size": 2.5, "Fuel_Type": "Gas", "Gear_Type": "Automatic",
     "Mileage": 85000, "Region": "Riyadh"},
    {"Make": "Toyota", "Type": "Land Cruiser", "Year": 2017, "Origin": "Gulf Arabic", "Color": "Black",
     "Options": "Full", "Engine_Size": 5.7, "Fuel_Type": "Gas", "Gear_Type": "Automatic",
     "Mileage": 160000, "Region": "Jeddah"},
    {"Make": "Hyundai", "Type": "Accent", "Year": 2012, "Origin": "Saudi", "Color": "Silver",
     "Options": "Standard", "Engine_Size": 1.6, "Fuel_Type": "Gas", "Gear_Type": "Manual",
     "Mileage": 240000, "Region": "Dammam"},
]


def show(title, r):
    print(f"\n--- {title}  [HTTP {r.status_code}]")
    print(json.dumps(r.json(), indent=2))


show("GET /", requests.get(f"{BASE}/"))
show("POST /predict/price", requests.post(f"{BASE}/predict/price", json=cars))
# for classification the target column is not sent; Price is optional
no_gear = [{k: v for k, v in c.items() if k != "Gear_Type"} | {"Price": p}
           for c, p in zip(cars, [70000, 150000, None])]
show("POST /predict/class?target=gear_type", requests.post(f"{BASE}/predict/class?target=gear_type", json=no_gear))
no_opt = [{k: v for k, v in c.items() if k != "Options"} for c in cars]
show("POST /predict/class?target=options", requests.post(f"{BASE}/predict/class?target=options", json=no_opt))
show("Bad request (missing Make)", requests.post(f"{BASE}/predict/price", json={"Year": 2018}))
