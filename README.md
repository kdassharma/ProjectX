# ProjectX
Embedded system to alert drowsy drivers using computer vision.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---
## Installation

Download facial detection model:
https://drive.google.com/file/d/1Z7ikr31fQJdv9ZsLB1zGF5D4YzsYMbtn/view?usp=sharing

## Usage

```shell
$ python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
$ python python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
```

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
