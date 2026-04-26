import pandas as pd
import requests
from geopy.geocoders import ArcGIS
from tqdm import tqdm


def get_altitude(latitude, longitude):
    # Define the Open-Elevation API endpoint
    url = f'https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}'

    # Send a GET request to the API endpoint
    response = requests.get(url)

    # Parse the JSON response
    data = response.json()

    # Extract altitude from the response
    altitude = data['results'][0]['elevation']

    return altitude


def get_location_data(address):
    # Initialize the ArcGIS geocoder
    geolocator = ArcGIS()

    # Retrieve location data
    location = geolocator.geocode(address,timeout=50*60)

    # Extract latitude and longitude
    latitude = location.latitude
    longitude = location.longitude

    # Retrieve altitude data using Open-Elevation API
    altitude = get_altitude(latitude, longitude)

    return latitude, longitude, altitude

df = pd.read_csv('./local_governments.csv')
municipalities = df['Local Level Name']
districts = df['District']
new_df = pd.DataFrame(columns=['Municipality', 'latitude', 'longitude', 'altitude'])
for i,munacipality in enumerate(municipalities):
    print(i)
    # Example usage

    address = f'{munacipality}, office, {districts[i]}, Nepal'
    # print(address)
    latitude, longitude, altitude = get_location_data(address)#timeout=10
    data = {
        'Municipality': munacipality,
        'latitude': latitude,
        'longitude': longitude,
        'altitude': altitude
    }
    # Append data to the DataFrame
    new_df = new_df.append(data, ignore_index=True)
new_df.to_csv('./municipality_geometry_753.csv', index=False)
print("Done !!!")
