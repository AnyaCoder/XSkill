"""Smoke-test an AutoDL OpenAI-compatible endpoint for EviLift requirements."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests


def _data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _request(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    endpoint = os.environ["REASONING_END_POINT"]
    payload: dict[str, Any] = {
        "model": os.environ["REASONING_MODEL_NAME"],
        "messages": messages,
        "temperature": 0,
        "max_tokens": 64,
    }
    if tools:
        payload["tools"] = tools
        payload["parallel_tool_calls"] = False
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {os.environ['REASONING_API_KEY']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=240,
    )
    response.raise_for_status()
    return response.json()


def _message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"Response has no choices: {json.dumps(response)[:500]}")
    return choices[0].get("message") or {}


def run_smoke_tests(images: list[Path]) -> dict[str, bool]:
    text_message = _message(_request([{"role": "user", "content": "Reply with OK only."}]))
    text_ok = bool(text_message.get("content"))

    multimodal_content: list[dict[str, Any]] = [
        {"type": "text", "text": "Briefly confirm that you received two images."}
    ]
    for image in images:
        multimodal_content.append({"type": "image_url", "image_url": {"url": _data_uri(image)}})
    image_message = _message(_request([{"role": "user", "content": multimodal_content}]))
    images_ok = bool(image_message.get("content"))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "zoom",
                "description": "Crop paired image regions for closer inspection.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]
    tool_message = _message(
        _request(
            [
                {
                    "role": "user",
                    "content": "You must call the zoom tool once with short Python crop code. Do not answer in text.",
                }
            ],
            tools=tools,
        )
    )
    tools_ok = bool(tool_message.get("tool_calls"))
    return {"text": text_ok, "two_images": images_ok, "tool_calls": tools_ok}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", required=True)
    args = parser.parse_args()
    if len(args.image) != 2:
        parser.error("exactly two --image arguments are required")
    results = run_smoke_tests(args.image)
    print(json.dumps(results, indent=2))
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

