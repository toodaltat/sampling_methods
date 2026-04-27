# sampling_methods

A Python project for measuring table occupancy in public spaces from video footage. The script uses YOLO object detection and a simple SORT-style tracker to identify people in each frame, estimate whether they are inside predefined table zones, and write occupancy data to a CSV file. The output can then be used for exploratory analysis of how environmental features such as shade, temperature, and distance from roads relate to seating behaviour.

## What this project does

This project processes a recorded video and at fixed time intervals, logs:
- timestamp
- occupancy count at each unique table
- distance from road
- whether the table is in shadow
- temperature

The main goal is to create a structured dataset that can be used to study public space use.

## How it works

For each frame in the video, the script:
1. detects people using YOLO
2. tracks detections across frames using a simple SORT-style tracker
3. checks whether each tracked person falls inside a manually defined table zone
4. counts the number of people in each zone
5. writes the results to a CSV file at the chosen logging interval
6. at chosen the chosen logging interval temperature data will be polled 

## Project structure

```text
sampling_methods/
├── 00_data/
│   └── video.mp4
├── 01_src/
│   ├── .env
│   ├── 00_point_finder.py
│   ├── 01_workbench.py
│   ├── 02_main.py
│   ├── sort.py
│   └── yolov8n.pt
├── 02_output/
│   └── occupancy_log.csv
├── 03_report/
│   ├── draft.txt
│   └── ProjectProposal.Rmd
├── 04_snapshotting/
│   ├── mount-data.service
│   ├── mount_data.sh
│   ├── README.md
│   └── recording.service
├── .gitignore
├── global.R
├── requirements.txt
└── README.md
```

## Setup

Begin with installing all required packages.

```commandline
pip install -r requirements.txt
```

Create a .env file in the project root and add your API key.

```text
METOCEAN_API_KEY=your_api_key
```

## Define table zones

Next step is to run "point_finder.py". This script looks in the data/ folder for a file named "video.mp4".

```commandline
python src/point_finder.py
```

Right click around where you'd like to define a table zone. 
When satisfied push the key "c" to get the list of coordinates and paste these coordinates into "src/main.py"

Repeat this process until you have defined all required zones for the footage.

## Checking footage

### workbench.py

This file is used for debugging and visual checking that the script is working correctly on the footage.

It is recommended to use this file before collecting the final data, so you can check.

* Table zones line up correctly
* people are being detected correctly
* there are no major artifacts in the footage that would affect the results.

The occupancy data will be written to,

```text
output/occupancy_log.csv
```