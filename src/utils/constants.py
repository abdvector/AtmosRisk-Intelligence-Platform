"""
constants.py

Static project constants used throughout ARIP.
"""

CITY_COORDINATES = {
    "Amritsar": [31.6340, 74.8723],
    "Ludhiana": [30.9010, 75.8573],
    "Chandigarh": [30.7333, 76.7794],
    "Delhi": [28.7041, 77.1025],
    "Gurgaon": [28.4595, 77.0266],
    "Noida": [28.5355, 77.3910],
    "Meerut": [28.9845, 77.7064],
    "Agra": [27.1767, 78.0081],
    "Kanpur": [26.4499, 80.3319],
    "Lucknow": [26.8467, 80.9462],
    "Varanasi": [25.3176, 82.9739],
    "Patna": [25.5941, 85.1376],
    "Gaya": [24.7914, 85.0002],
    "Muzaffarpur": [26.1209, 85.3647],
    "Kolkata": [22.5726, 88.3639],
    "Asansol": [23.6739, 86.9524],
    "Siliguri": [26.7271, 88.3953],
    "Jaipur": [26.9124, 75.7873],
    "Gwalior": [26.2183, 78.1828],
    "Bhopal": [23.2599, 77.4126],
}

POLLUTANT_FEATURES = [
    "PM25",
    "PM10",
    "NO2",
    "SO2"
]

HIGH_RISK_CITIES = [
    "Delhi",
    "Noida",
    "Gurgaon",
    "Patna"
]