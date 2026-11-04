import json
import sys
import os

# Minimum acceptable accuracy
MIN_ACCURACY = 0.85

# Check whether metrics file exists
if not os.path.exists("metrics.json"):
    print("ERROR: metrics.json file not found")
    sys.exit(1)

# Read model accuracy
with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print("Model Accuracy:", accuracy)
print("Minimum Required Accuracy:", MIN_ACCURACY)

# Check quality gate
if accuracy >= MIN_ACCURACY:
    print("QUALITY GATE PASSED")
    sys.exit(0)
else:
    print("QUALITY GATE FAILED")
    print("Model accuracy is below 85%")
    sys.exit(1)
