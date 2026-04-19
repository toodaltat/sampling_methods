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

## Project structure

```text
sampling_methods/
├── data/
│   └── video.mp4
├── output/
│   └── occupancy_log.csv
├── src/
│   ├── main_script.py
│   ├── sort.py
│   └── point_finder.py
├── .env
├── requirements.txt
└── README.md