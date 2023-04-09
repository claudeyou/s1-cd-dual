"""
Created on Fri Mar 12 19:06:01 2023

@author: hoyeong

### Download Earth Engine Images to Drive ###
"""

#%% Importing Library

import time
import ee

ee.Authenticate()
ee.Initialize()

#%% geoJSON

# https://geojson.io/#map=2/0/20 (Control(CMD) + Click)
geoJSON = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              -98.2122802734375,
              49.769291532628515
            ],
            [
              -98.00559997558594,
              49.769291532628515
            ],
            [
              -98.00559997558594,
              49.88578690918283
            ],
            [
              -98.2122802734375,
              49.88578690918283
            ],
            [
              -98.2122802734375,
              49.769291532628515
            ]
          ]
        ]
      }
    }
  ]
}


#%% Coords, Area of Interest

coords = geoJSON['features'][0]['geometry']['coordinates']
aoi = ee.Geometry.Polygon(coords)

#%% Find filter parameters

# find orbitProperties_pass:
# find relativeOrbitNumber_start:
# find sliceNumber: 
prm = ee.Image(ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') 
                       .filterBounds(aoi) 
                       .filterDate(ee.Date('2021-04-01'), ee.Date('2021-09-30')) 
                       .sort('system:time_start'))

#prm.getInfo()

#%% Image Collection
'''Consider sliceNumber'''
im_coll = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
                .filterBounds(aoi)
                .filterDate(ee.Date('2018-08-01'),ee.Date('2018-08-31'))
                .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                .filter(ee.Filter.eq('relativeOrbitNumber_start', 136))
                # .filter(ee.Filter.eq('sliceNumber', 3))
                .first()
                .clip(aoi))
                # .sort('system:time_start'))

#%% Creating Date String
acq_times = im_coll.aggregate_array('system:time_start').getInfo()
date_list = [time.strftime('%x', time.gmtime(acq_time/1000)) for acq_time in acq_times]
new_date_list = [date.replace('/', '_') for date in date_list] #mm_dd_yy

#%% Export Images to Google Drive
def export_to_drive(image, index):
    image_name = f'S1_GRD_{index}'
    
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=image_name,
        folder="S1_Seoul", #Your Google Drive folder
        fileNamePrefix=image_name,
        region=aoi,
        scale=10,
        fileFormat="GeoTIFF",
        maxPixels=1e13
    )
    
    task.start()
    
    print(f'Exporting {image_name} to Google Drive...')
    
    #Download Status
    while task.status()['state'] in ['READY', 'RUNNING']:
        print('Task status:', task.status())
        time.sleep(10)  # Wait for 15 seconds before checking the status again

    print('Task status:', task.status())
    
#%% Export all images of a collection

image_list = im_coll.toList(im_coll.size())

for i in range(image_list.size().getInfo()):
    img = ee.Image(image_list.get(i))
    success = False
    while not success:
        try:
            export_to_drive(img, new_date_list[i])
            success = True
        except Exception as e:
            print(f"Error occurred while exporting image {new_date_list[i]}: {e}")

