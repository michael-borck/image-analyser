# Basic usage

`image-analyser` runs static image analysis (metadata, quality, optional OCR and captioning) for the analyser family.

## Install

```bash
pip install image-analyser
```

Optional extras: `[ocr]` for text extraction, `[ml]` for local captioning, `[api]` for vision-model captioning (or `[all]`).

## CLI

```bash
image-analyser path/to/photo.jpg --json
```

## Python

```python
from image_analyser import ImageAnalyser

analyser = ImageAnalyser()
result = analyser.analyse("path/to/photo.jpg")  # AnalysisResult (pydantic)
print(result.model_dump_json(indent=2))
```

## HTTP

```bash
image-analyser serve
curl -F file=@path/to/photo.jpg http://localhost:8006/analyse
```
